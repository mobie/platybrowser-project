#!/usr/bin/env -S uv run
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "zarr<3",
#   "fsspec",
#   "pandas",
#   "dask",
#   "numpy"
# ]
# ///

import zarr
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import warnings
import os
import time
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Configuration ---
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/cros/trace_merging")
PATH_TSV = os.path.join(SCRATCH_DIR, 'default.tsv')
OUTPUT_N5 = os.path.join(SCRATCH_DIR, 'sbem-6dpf-1-whole-combined-traces.n5')

n5_paths = {
    "nuclei": os.path.join(SCRATCH_DIR, "sbem-6dpf-1-whole-segmented-nuclei.n5"),
    "kevin_traces": os.path.join(SCRATCH_DIR, "sbem-6dpf-1-whole-traces.n5"),
    "david_traces": os.path.join(SCRATCH_DIR, "sbem-6dpf-1-whole-traces-david.n5") # Added David back
}

# --- Custom SLURM Progress Bar with Time ---
class SlurmProgress(Callback):
    def __init__(self, step_name):
        self.step_name = step_name
        self.last_percent = -1
        self.start_time = time.time()

    def _posttask(self, key, result, dsk, state, worker_id):
        finished = len(state.get('finished', []))
        total = sum(len(v) for v in state.values())
        if total > 0:
            percent = int(100 * finished / total)
            if percent % 2 == 0 and percent > self.last_percent:
                elapsed = int(time.time() - self.start_time)
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                clock = datetime.now().strftime('%H:%M:%S')
                print(f"[{clock}] [{self.step_name}] Progress: {percent}% ({finished}/{total} chunks) | Elapsed: {h:02d}:{m:02d}:{s:02d}", flush=True)
                self.last_percent = percent

def create_lut(mapping_series, max_input_id):
    lut = np.zeros(int(max_input_id) + 1, dtype=np.int32)
    for old_id, new_id in mapping_series.items():
        if old_id > 0 and new_id > 0:
            lut[int(old_id)] = int(new_id)
    return lut

def apply_lut_to_chunk(chunk, lut):
    chunk_int = chunk.astype(np.int32)
    valid_mask = (chunk_int > 0) & (chunk_int < len(lut))
    result = np.zeros(chunk.shape, dtype=np.uint16)
    result[valid_mask] = lut[chunk_int[valid_mask]].astype(np.uint16)
    return result

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading mapping table from {PATH_TSV}...\n")
    df = pd.read_csv(PATH_TSV, sep='\t')
    
    # 1. BUILD FAST LOOKUP TABLES (Kevin & David)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building Look-Up Tables...")
    
    max_nuclei_id = df['label_id'].max()
    lut_nuclei = create_lut(pd.Series(df['label_id'].values, index=df['label_id'].values), max_nuclei_id)
    
    kevin_df = df[df['kevin_head_traces_id'] > 0]
    max_kevin_id = kevin_df['kevin_head_traces_id'].max() if not kevin_df.empty else 0
    lut_kevin = create_lut(kevin_df.set_index('kevin_head_traces_id')['label_id'], max_kevin_id)

    david_df = df[df['david_motorneuron_2nd_segment_traces_id'] > 0]
    max_david_id = david_df['david_motorneuron_2nd_segment_traces_id'].max() if not david_df.empty else 0
    lut_david = create_lut(david_df.set_index('david_motorneuron_2nd_segment_traces_id')['label_id'], max_david_id)

    # 2. PREPARE OUTPUT STORE
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing output N5 structure at {OUTPUT_N5}...")
    out_store = zarr.N5FSStore(OUTPUT_N5)
    out_root = zarr.group(store=out_store, overwrite=True)

    temp_store = zarr.N5FSStore(n5_paths['nuclei'])
    temp_root = zarr.open(temp_store, mode='r')

    # Copy Hierarchy & Inject safe MoBIE/BDV metadata
    for path in ['', 'setup0', 'setup0/timepoint0']:
        src_group = zarr.open(temp_store, path=path, mode='r')
        dst_group = out_root.require_group(path)
        safe_attrs = {k: v for k, v in src_group.attrs.asdict().items() if k in ['n5', 'downsamplingFactors', 'path']}
        if path == 'setup0':
            safe_attrs['dataType'] = 'uint16'
        dst_group.attrs.update(safe_attrs)

    # ---------------------------------------------------------
    # 3. HELPER: SMART TRACE LOADER
    # ---------------------------------------------------------
    def load_trace_scale(dataset_key, scale_name, lut, target_shape, target_chunks):
        """Loads a trace scale with fallback to dynamic downsampling if missing."""
        try:
            raw = da.from_zarr(zarr.N5FSStore(n5_paths[dataset_key]), component=f'setup0/timepoint0/{scale_name}')
            if raw.shape != target_shape:
                raise ValueError(f"Shape mismatch: Trace {raw.shape} vs Nuclei {target_shape}")
            print(f"  -> Found perfect pre-computed match for {dataset_key} at {scale_name}.")
            return raw.map_blocks(apply_lut_to_chunk, lut=lut, dtype=np.uint16)
        except Exception as e:
            print(f"  -> {dataset_key} missing/mismatched at {scale_name}. Using Max-Pooling fallback...")
            
            src_store = zarr.N5FSStore(n5_paths[dataset_key])
            src_zarr = zarr.open(src_store, mode='r')
            s0_raw = da.from_zarr(src_zarr['setup0/timepoint0/s0'])
            s0_mapped = s0_raw.map_blocks(apply_lut_to_chunk, lut=lut, dtype=np.uint16)
            
            step_z = max(1, round(s0_raw.shape[0] / target_shape[0]))
            step_y = max(1, round(s0_raw.shape[1] / target_shape[1]))
            step_x = max(1, round(s0_raw.shape[2] / target_shape[2]))
            
            # Max-Pooling: Preserves thin traces instead of erasing them
            sampled = da.coarsen(np.max, s0_mapped, {0: step_z, 1: step_y, 2: step_x}, trim_excess=True)
            
            # Slice if slightly too large due to rounding
            sampled = sampled[:target_shape[0], :target_shape[1], :target_shape[2]]
            
            # Pad with zeros if slightly too small
            pad_z = max(0, target_shape[0] - sampled.shape[0])
            pad_y = max(0, target_shape[1] - sampled.shape[1])
            pad_x = max(0, target_shape[2] - sampled.shape[2])
            
            if pad_z > 0 or pad_y > 0 or pad_x > 0:
                sampled = da.pad(sampled, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                
            return sampled.rechunk(target_chunks)

    # ---------------------------------------------------------
    # 4. HELPER: MERGE A SPECIFIC SCALE
    # ---------------------------------------------------------
    def merge_and_write_scale(scale_name):
        print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] Processing {scale_name} ---")
        
        # 1. Load the pre-computed template from Nuclei
        nuc_sk_raw = da.from_zarr(zarr.N5FSStore(n5_paths['nuclei']), component=f'setup0/timepoint0/{scale_name}')
        nuc_mapped = nuc_sk_raw.map_blocks(apply_lut_to_chunk, lut=lut_nuclei, dtype=np.uint16)
        
        temp_sk = temp_root[f'setup0/timepoint0/{scale_name}']
        
        # 2. Smart-Load Both Traces
        kev_mapped = load_trace_scale('kevin_traces', scale_name, lut_kevin, nuc_sk_raw.shape, temp_sk.chunks)
        dav_mapped = load_trace_scale('david_traces', scale_name, lut_david, nuc_sk_raw.shape, temp_sk.chunks)
        
        # 3. Stack layers: Kevin overwrites David, both overwrite Nuclei
        traces_combined = da.where(kev_mapped > 0, kev_mapped, dav_mapped)
        combined_sk = da.where(traces_combined > 0, traces_combined, nuc_mapped)
        
        combined_sk = combined_sk.rechunk(temp_sk.chunks)
        
        # 4. Initialize and write the output level
        out_sk = zarr.create(
            store=out_store,
            path=f'setup0/timepoint0/{scale_name}',
            shape=temp_sk.shape,
            chunks=temp_sk.chunks,
            dtype=np.uint16,
            compressor=zarr.GZip(level=5),
            fill_value=0,
            overwrite=True
        )

        with SlurmProgress(f"{scale_name} Merging"):
            da.store(combined_sk, out_sk)
            
        safe_sk_attrs = {k: v for k, v in temp_sk.attrs.asdict().items() if k in ['dimensions', 'blockSize', 'compression', 'downsamplingFactors']}
        safe_sk_attrs['dataType'] = 'uint16'
        out_sk.attrs.update(safe_sk_attrs)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully wrote {scale_name}.")

    # ---------------------------------------------------------
    # 5. EXECUTE LEVEL-BY-LEVEL MERGE
    # ---------------------------------------------------------
    
    # Do the base resolution first
    merge_and_write_scale('s0')
    
    # Find all remaining pre-computed scales (s1, s2, s3, etc.)
    scales = sorted([k for k in temp_root['setup0/timepoint0'].keys() if k.startswith('s') and k != 's0'])
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found pre-computed pyramids to merge: {scales}")
    
    # Do the pyramids
    for scale in scales:
        merge_and_write_scale(scale)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Pipeline complete! Full N5 pyramid merged perfectly.")

if __name__ == "__main__":
    main()