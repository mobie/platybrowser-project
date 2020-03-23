import json
from concurrent import futures
from tqdm import tqdm

import nifty.tools as nt
import numpy as np
import z5py
from heimdall import view, to_source
from subdivide_for_proofreading import get_blocking, tentative_block_shape
from elf.io.label_multiset_wrapper import LabelMultisetWrapper

TMP_PATH = './data.n5'


# debug check list:
# - preprocessing:
# -- exported segmentation looks correct
# -- block partition looks correct
# -- checking block coverage is correct
# -- assignment coverage is correct


def debug_preprocessing_seg():
    p_seg = './data.n5'
    k_seg = 'volumes/segmentation'

    halo = [50, 512, 512]
    with z5py.File(p_seg, 'r') as f:
        ds = f[k_seg]
        ds.n_threads = 8
        shape = ds.shape
        center = [sh // 2 for sh in shape]
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        seg = ds[bb]

    p_raw = '../../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    k_raw = 'setup0/timepoint0/s1'
    with z5py.File(p_raw, 'r') as f:
        ds = f[k_raw]
        ds.n_threads = 8
        print(ds.shape)
        raw = ds[bb]

    view(to_source(raw),
         to_source(seg))


def debug_preprocessing_blocks():
    p_blocks = './data.n5'
    k_blocks = 'labels_for_subdivision'

    halo = [50, 512, 512]
    with z5py.File(p_blocks, 'r') as f:
        ds = f[k_blocks]
        ds.n_threads = 8
        shape = ds.shape
        center = [sh // 2 for sh in shape]
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        blocks = ds[bb]

    p_raw = '../../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    k_raw = 'setup0/timepoint0/s3'
    with z5py.File(p_raw, 'r') as f:
        ds = f[k_raw]
        ds.n_threads = 8
        raw = ds[bb]

    view(raw, blocks)


def debug_label_to_block_mapping():
    with open('./labels_to_blocks.json', 'r') as f:
        label_block_mapping = json.load(f)
    label_block_mapping = {int(k): v for k, v in label_block_mapping.items()}
    print(len(label_block_mapping))

    scale = 2
    block_shape = tentative_block_shape(scale)
    shape, blocking = get_blocking(scale, block_shape)

    scale_factor = 4
    p = './data.n5'
    # k_blocks = 'labels_for_subdivision'
    k_seg = 'volumes/segmentation'

    f = z5py.File(p, 'r')
    ds_seg = f[k_seg]
    # ds_blocks = f[k_blocks]

    check_block_shape = ds_seg.chunks

    def check_block(block_id):
        labels = label_block_mapping[block_id]
        if len(labels) == 0:
            return True

        block = blocking.getBlock(block_id - 1)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        roi_begin = [b.start * scale_factor for b in bb]
        roi_end = [b.stop * scale_factor for b in bb]
        blocking_seg = nt.blocking(roi_begin, roi_end, check_block_shape)

        this_ids = set()
        for check_block_id in range(blocking_seg.numberOfBlocks):
            check_block = blocking_seg.getBlock(check_block_id)
            check_bb = tuple(slice(beg, end) for beg, end in zip(check_block.begin, check_block.end))
            seg = ds_seg[check_bb]
            ids = np.unique(seg)
            this_ids.update(ids.tolist())

        if len(set(labels) - this_ids) > 0:
            return False
        else:
            return True

    # print(check_block(8))
    n_threads = 32
    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(tp.map(check_block, label_block_mapping.keys())))

    print(sum(results), "/", len(results))
    results = {block_id: res for block_id, res in zip(label_block_mapping.keys(), results)}
    with open('debug_block_res.json', 'w') as f:
        json.dump(results, f)


def debug_assignments_and_block_labels():
    with z5py.File(TMP_PATH, 'r') as f:
        assignments = f['node_labels/fragment_segment_assignment'][:]
    print(assignments.shape)
    label_ids = np.unique(assignments[:, 1])
    print(len(label_ids))

    with open('./labels_to_blocks.json', 'r') as f:
        label_block_mapping = json.load(f)
    mapped_ids = np.array([mid for ids in label_block_mapping.values() for mid in ids])
    print(len(mapped_ids))

    id_diff = np.setdiff1d(mapped_ids, label_ids)
    print(id_diff.shape)


def check_mapped_watersheds():
    with z5py.File(TMP_PATH, 'r') as f:
        assignments = f['node_labels/fragment_segment_assignment'][:]

    with open('./labels_to_blocks.json', 'r') as f:
        labels_to_blocks = json.load(f)
    labels_to_blocks = {int(k): v for k, v in labels_to_blocks.items()}

    with open('./rois_to_blocks.json', 'r') as f:
        rois_to_blocks = json.load(f)
    rois_to_blocks = {int(k): v for k, v in rois_to_blocks.items()}

    p = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    k_seg = 'volumes/paintera/proofread_cells_multiset/data/s0'

    f = z5py.File(p, 'r')
    ds_seg = f[k_seg]
    check_block_shape = ds_seg.chunks
    ds_seg = LabelMultisetWrapper(ds_seg)

    def check_block(block_id):
        labels = labels_to_blocks[block_id]
        if len(labels) == 0:
            return []

        assignment_mask = np.isin(assignments[:, 1], labels)
        assert assignment_mask.sum() > 0
        block_assignments = assignments[assignment_mask]
        block_ws_labels = block_assignments[:, 0]

        roi_begin, roi_end = rois_to_blocks[block_id]
        blocking_seg = nt.blocking(roi_begin, roi_end, check_block_shape)

        this_ids = set()
        for check_block_id in range(blocking_seg.numberOfBlocks):
            check_block = blocking_seg.getBlock(check_block_id)
            check_bb = tuple(slice(beg, end) for beg, end in zip(check_block.begin, check_block.end))
            seg = ds_seg[check_bb]
            ids = np.unique(seg)
            this_ids.update(ids.tolist())

        diff = set(block_ws_labels) - this_ids
        return list(diff)

    n_threads = 32
    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(tp.map(check_block, labels_to_blocks.keys())))

    print("Calculation done")
    results = {block_id: res for block_id, res in zip(labels_to_blocks.keys(), results)}
    with open('debug_block_ws_res.json', 'w') as f:
        json.dump(results, f)


def check_selected_mapping():
    with z5py.File(TMP_PATH, 'r') as f:
        assignments = f['node_labels/fragment_segment_assignment'][:]
        morpho = f['morphology'][:]

    with open('./labels_to_blocks.json', 'r') as f:
        labels_to_blocks = json.load(f)
    labels_to_blocks = {int(k): v for k, v in labels_to_blocks.items()}

    ps = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    pr = '../../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    k_raw = 'setup0/timepoint0/s1'
    k_seg = 'volumes/paintera/proofread_cells_multiset/data/s0'

    fs = z5py.File(ps, 'r')
    ds_seg = fs[k_seg]
    ds_seg = LabelMultisetWrapper(ds_seg)

    fr = z5py.File(pr, 'r')
    ds_raw = fr[k_raw]
    ds_raw.n_threads = 8

    def check_objects(block_id, n_objects=5):
        labels = labels_to_blocks[block_id]
        if len(labels) == 0:
            return

        sampled_ids = np.random.choice(labels, n_objects, False)
        for label_id in sampled_ids:
            assignment_mask = assignments[:, 1] == label_id
            assert assignment_mask.sum() > 0
            this_assignments = assignments[assignment_mask]
            this_ws_labels = this_assignments[:, 0]

            roi_start = morpho[label_id, 5:8].astype('uint64')
            roi_stop = morpho[label_id, 8:11].astype('uint64') + 1
            bb = tuple(slice(sta, sto) for sta, sto in zip(roi_start.tolist(),
                                                           roi_stop.tolist()))
            ws = ds_seg[bb]
            ws_mask = np.isin(ws, this_ws_labels)
            print("Loading object", label_id)
            print("N-foreground:", ws_mask.sum())
            ws[~ws_mask] = 0

            raw = ds_raw[bb]
            view(raw, ws)

    check_objects(8)


if __name__ == '__main__':
    # debug_preprocessing_seg()
    # debug_preprocessing_blocks()
    # debug_label_to_block_mapping()
    # debug_assignments_and_block_labels()
    # check_mapped_watersheds()
    # check_mapped_watersheds()
    check_selected_mapping()
