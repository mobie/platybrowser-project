#!/usr/bin/env -S uv run
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Generate per-subtype nuclei probability N5 images.

For each detailed cell type (subtype) column in the detailed cluster
probability table, repaint the nuclei segmentation mask: every nucleus voxel
keeps its geometry but its value becomes round(p * 1000) for that subtype.
Background stays 0. Output N5s mirror the mask's pyramid exactly (levels,
shapes, chunking, resolution/downsamplingFactors/offset attributes) and use
gzip + fillvalue 0 so images are sparse and small.

Usage:
    ./generate_nuclei_proba_images.py --mask <nuclei.n5> --table <tsv> \
        --stage-dir <dir> --local-xml-dir <dir> [--subtypes a,b,c] \
        [--workers N] [--gzip-level N]
"""

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import z5py

SUBSTYPE_PREFIXES = ("clade", "nocladesub")
EXCLUDED_COLUMNS = {"label_id", "zero", "autofluorescence", "most probable cluster"}
SCALE = 1000


def safe_name(subtype):
    """Filesystem/URL-safe artifact name for a score column.

    Spaces and slashes are replaced by underscores: MoBIE fetches source XMLs
    from raw.githubusercontent.com without URL-encoding, so relative paths must
    not contain spaces (HTTP 400).
    """
    return subtype.replace("/", "_").replace(" ", "_")
DEFAULT_XML_TEMPLATE = Path(__file__).resolve().parent.parent / \
    "data" / "platybrowser_6dpf" / "images" / "local" / \
    "sbem-6dpf-1-whole-segmented-nuclei.xml"

# Process-pool shared state. Set in write_outputs just before the pool is
# created; the fork start method (Linux default) inherits it copy-on-write,
# so the ~274 uint16 value tables are never pickled per task.
_POOL_MASK_PATH = None
_POOL_STAGE_DIR = None
_POOL_VALUE_TABLES = None


def read_subtype_columns(tsv_path):
    """Return subtype probability columns in table order (clade*/nocladesub* only)."""
    with open(tsv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f, delimiter="\t"))
    return [c for c in header
            if c.startswith(SUBSTYPE_PREFIXES) and c not in EXCLUDED_COLUMNS]


def read_probabilities(tsv_path, subtype):
    """Return {label_id: probability} for one subtype column."""
    probs = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = int(float(row["label_id"]))
            probs[label] = float(row[subtype])
    return probs


def build_value_table(probs, scale=SCALE):
    """uint16 lookup table indexed by label: value = round(p * scale), label 0 -> 0."""
    max_label = max(probs.keys(), default=0)
    table = np.zeros(max_label + 1, dtype=np.uint16)
    for label, p in probs.items():
        table[label] = int(round(p * scale))
    return table


def mirror_level_info(mask_path):
    """Introspect the mask pyramid: level name, shape, chunks, attrs per s-level."""
    with z5py.File(str(mask_path), "r") as f:
        tp = f["setup0/timepoint0"]
        levels = []
        for key in sorted(tp.keys()):
            ds = tp[key]
            levels.append({
                "name": key,
                "shape": tuple(ds.shape),
                "chunks": tuple(ds.chunks),
                "attrs": dict(ds.attrs),
            })
    return levels


def mirror_group_attrs(mask_path):
    """Copy the mask's setup0 and timepoint0 group attributes (dataType,
    multiScale, resolution, per-level downsamplingFactors)."""
    with z5py.File(str(mask_path), "r") as f:
        return {
            "setup0": dict(f["setup0"].attrs),
            "timepoint0": dict(f["setup0/timepoint0"].attrs),
        }


def relabel_block(mask_block, value_table):
    """Map a uint32 label block through the value table (labels beyond table -> 0)."""
    out = np.zeros(mask_block.shape, dtype=np.uint16)
    idx = mask_block <= value_table.shape[0] - 1
    out[idx] = value_table[mask_block[idx].astype(np.intp)]
    return out


def _iter_block_slices(shape, chunks):
    """Yield every block slice (z, y, x) of an array in chunk-aligned order."""
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                yield (slice(z0, min(z0 + chunks[0], shape[0])),
                       slice(y0, min(y0 + chunks[1], shape[1])),
                       slice(x0, min(x0 + chunks[2], shape[2])))


def _fill_blocks_sequential(mask_path, value_tables, stage_dir, levels):
    """Single-process block-fill: every non-zero mask block relabeled per subtype."""
    with z5py.File(str(mask_path), "r") as mask_f:
        tp = mask_f["setup0/timepoint0"]
        for level in levels:
            name, shape, chunks = level["name"], level["shape"], level["chunks"]
            mask_ds = tp[name]
            out_dss = {
                subtype: z5py.File(str(stage_dir / f"{safe_name(subtype)}_proba.n5"), "a")[
                    "setup0/timepoint0"][name]
                for subtype in value_tables
            }
            for sl in _iter_block_slices(shape, chunks):
                block = mask_ds[sl]
                if not block.any():
                    continue
                for subtype, value_table in value_tables.items():
                    out_dss[subtype][sl] = relabel_block(block, value_table)


def _partition(seq, n):
    """Split seq into <= n contiguous, disjoint chunks covering it exactly."""
    n = min(n, len(seq))
    k, m = divmod(len(seq), n)
    chunks = []
    start = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        chunks.append(seq[start:start + size])
        start += size
    return chunks


def _fill_level_blocks(task):
    """Worker entry: fill a chunk of block slices for one pyramid level.

    Opens the level's mask dataset and the output datasets once, streams the
    block-slice list (skip all-zero, else relabel + write per subtype), closes.
    The value tables are read from the module-level pool globals (inherited
    copy-on-write via fork, never pickled).
    """
    level, block_slices = task
    name = level["name"]
    mask_f = z5py.File(str(_POOL_MASK_PATH), "r")
    mask_ds = mask_f["setup0/timepoint0"][name]
    out_fs = {}
    out_fs = {
        subtype: z5py.File(str(_POOL_STAGE_DIR / f"{safe_name(subtype)}_proba.n5"), "a")
        for subtype in _POOL_VALUE_TABLES
    }
    out_dss = {s: out_fs[s]["setup0/timepoint0"][name] for s in out_fs}
    try:
        for sl in block_slices:
            block = mask_ds[sl]
            if not block.any():
                continue
            for subtype, value_table in _POOL_VALUE_TABLES.items():
                out_dss[subtype][sl] = relabel_block(block, value_table)
    finally:
        mask_f.close()
        for f in out_fs.values():
            f.close()


def _fill_blocks_parallel(mask_path, value_tables, stage_dir, levels, n_workers):
    """Multi-process block-fill: one task per (level, block chunk)."""
    global _POOL_MASK_PATH, _POOL_STAGE_DIR, _POOL_VALUE_TABLES
    _POOL_MASK_PATH = Path(mask_path)
    _POOL_STAGE_DIR = Path(stage_dir)
    _POOL_VALUE_TABLES = value_tables
    try:
        tasks = []
        for level in levels:
            block_slices = list(_iter_block_slices(level["shape"], level["chunks"]))
            for chunk in _partition(block_slices, n_workers):
                if chunk:
                    tasks.append((level, chunk))
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            for _ in pool.map(_fill_level_blocks, tasks):
                pass
    finally:
        _POOL_MASK_PATH = None
        _POOL_STAGE_DIR = None
        _POOL_VALUE_TABLES = None


def write_outputs(mask_path, value_tables, stage_dir, levels, group_attrs,
                  n_workers=1, gzip_level=1):
    """Write one uint16 N5 per subtype into stage_dir, mirroring the mask pyramid.

    Single block-wise pass over each mask level. All-zero blocks are skipped;
    the N5 fillvalue 0 covers them, keeping the outputs sparse and small.

    The first pass (N5 creation) is always sequential; the block-fill second
    pass runs in a process pool when n_workers > 1 (one task per level + block
    chunk, fork-inherited value tables). gzip_level sets the N5 compression
    level used at creation. With n_workers <= 1 the block-fill runs in-process
    and produces the same values and attrs as the pool path.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for subtype in value_tables:
        out_path = stage_dir / f"{safe_name(subtype)}_proba.n5"
        if out_path.exists():
            shutil.rmtree(out_path)
        with z5py.File(str(out_path), "a") as f:
            setup_group = f.create_group("setup0")
            for k, v in group_attrs.get("setup0", {}).items():
                setup_group.attrs[k] = v
            tp_group = setup_group.create_group("timepoint0")
            for k, v in group_attrs.get("timepoint0", {}).items():
                tp_group.attrs[k] = v
            for level in levels:
                ds = tp_group.create_dataset(
                    level["name"],
                    shape=level["shape"],
                    chunks=level["chunks"],
                    dtype="uint16",
                    compression="gzip",
                    level=gzip_level,
                    fillvalue=0,
                )
                for k, v in level["attrs"].items():
                    ds.attrs[k] = v
        written.append(out_path)

    if n_workers > 1:
        _fill_blocks_parallel(mask_path, value_tables, stage_dir, levels, n_workers)
    else:
        _fill_blocks_sequential(mask_path, value_tables, stage_dir, levels)
    return written


def write_local_xmls(subtypes, local_xml_dir, stage_dir, xml_template):
    """Write {subtype}_proba.xml copies of the template with name + n5 path updated."""
    local_xml_dir = Path(local_xml_dir)
    stage_dir = Path(stage_dir)
    local_xml_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for subtype in subtypes:
        safe = safe_name(subtype)
        name = f"{safe}_proba"
        tree = ET.parse(xml_template)
        root = tree.getroot()
        for name_el in root.iter("name"):
            name_el.text = name
        n5_el = root.find(".//ImageLoader/n5")
        n5_el.set("type", "relative")
        n5_el.text = os.path.relpath(stage_dir / f"{name}.n5", local_xml_dir).replace("\\", "/")
        out_path = local_xml_dir / f"{name}.xml"
        ET.indent(root, space="  ")
        tree.write(out_path, encoding="utf-8", xml_declaration=False)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n")
        written.append(out_path)
    return written


def dir_size(path):
    """Recursive byte count of all files under path."""
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def report_sizes(stage_dir, mask_path, subtypes):
    """Per-subtype: on-disk bytes of the generated N5 vs the mask N5, and ratio."""
    mask_total = dir_size(mask_path)
    rows = []
    for subtype in subtypes:
        n5 = Path(stage_dir) / f"{safe_name(subtype)}_proba.n5"
        total = dir_size(n5) if n5.is_dir() else 0
        rows.append({
            "subtype": subtype,
            "file_size_bytes": total,
            "mask_size_bytes": mask_total,
            "ratio": (total / mask_total) if mask_total else 0.0,
        })
    return rows


def print_report(rows):
    print(f"{'subtype':<24}{'file_bytes':>14}{'mask_bytes':>14}{'ratio':>12}")
    for r in rows:
        print(f"{r['subtype']:<24}{r['file_size_bytes']:>14}"
              f"{r['mask_size_bytes']:>14}{r['ratio']:>12.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True,
                   help="Path to the nuclei segmentation N5 (must contain setup0/timepoint0/s*).")
    p.add_argument("--table", required=True,
                   help="Path to detailed_cell_types_cluster_probability.tsv")
    p.add_argument("--stage-dir", required=True,
                   help="Dir for generated {subtype}_proba.n5 files (gitignored, e.g. tmp_celltype_proba)")
    p.add_argument("--local-xml-dir", required=True,
                   help="Dir for local {subtype}_proba.xml files (repo images/local)")
    p.add_argument("--xml-template", default=str(DEFAULT_XML_TEMPLATE))
    p.add_argument("--subtypes", default=None,
                   help="Comma-separated subtypes (default: all from the table)")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes for the block-fill pass (default: 1 = "
                        "sequential). Peak open file descriptors ~= workers x "
                        "(number of subtypes + 1); very large worker counts need "
                        "a high `ulimit -n`.")
    p.add_argument("--gzip-level", type=int, default=1,
                   help="Gzip compression level for the output N5s (default: 1)")
    p.add_argument("--report-json", default=None,
                   help="Write the size report to this JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    subtypes = args.subtypes.split(",") if args.subtypes else read_subtype_columns(args.table)
    if not subtypes:
        sys.exit("No subtypes selected")
    levels = mirror_level_info(args.mask)
    group_attrs = mirror_group_attrs(args.mask)
    value_tables = {
        subtype: build_value_table(read_probabilities(args.table, subtype))
        for subtype in subtypes
    }
    write_outputs(args.mask, value_tables, Path(args.stage_dir), levels, group_attrs,
                  n_workers=args.workers, gzip_level=args.gzip_level)
    write_local_xmls(subtypes, Path(args.local_xml_dir), Path(args.stage_dir),
                     Path(args.xml_template))
    rows = report_sizes(Path(args.stage_dir), Path(args.mask), subtypes)
    print_report(rows)
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
    total = sum(r["file_size_bytes"] for r in rows)
    print(f"\nWrote {len(subtypes)} images + local XMLs. "
          f"Total output bytes: {total} "
          f"(mask: {rows[0]['mask_size_bytes'] if rows else 0})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
