import os
from math import ceil, floor

import numpy as np
import vigra
from elf.io import open_file


def export_node_labels(path, in_key, out_key, project_folder):
    with open_file(path, 'r') as f:
        ds = f[in_key]
        node_labels = ds[:]

    result_folder = os.path.join(project_folder, 'results')
    res_files = os.listdir(result_folder)

    print("Applying changes for", len(res_files), "resolved objects")

    id_offset = int(node_labels.max()) + 1
    for resf in res_files:
        seg_id = int(os.path.splitext(resf)[0])
        resf = os.path.join(result_folder, resf)
        res = np.load(resf)

        this_ids, this_labels = res['node_ids'], res['node_labels']
        assert len(this_ids) == len(this_labels)
        this_ids_exp = np.where(node_labels == seg_id)[0]
        assert np.array_equal(np.sort(this_ids), this_ids_exp)

        this_labels += id_offset
        node_labels[this_ids] = this_labels
        id_offset = int(this_labels.max()) + 1

    with open_file(path) as f:
        chunks = (min(int(1e6), len(node_labels)),)
        ds = f.require_dataset(out_key, compression='gzip', dtype=node_labels.dtype,
                               chunks=chunks, shape=node_labels.shape)
        ds[:] = node_labels


def to_paintera_format(in_path, in_key, out_path, out_key):
    with open_file(in_path, 'r') as f:
        node_labels = f[in_key][:]
    node_labels = vigra.analysis.relabelConsecutive(node_labels, start_label=1, keep_zeros=True)[0]

    n_ws = len(node_labels)
    ws_ids = np.arange(n_ws, dtype='uint64')
    assert len(ws_ids) == len(node_labels)

    seg_ids, seg_counts = np.unique(node_labels, return_counts=True)
    trivial_segments = seg_ids[seg_counts == 1]
    trivial_mask = np.in1d(node_labels, trivial_segments)

    ws_ids = ws_ids[~trivial_mask]
    node_labels = node_labels[~trivial_mask]

    node_labels[node_labels != 0] += n_ws

    max_id = node_labels.max()
    print("new max id:", max_id)

    paintera_labels = np.concatenate([ws_ids[:, None], node_labels[:, None]], axis=1).T
    print(paintera_labels.shape)

    with open_file(out_path) as f:
        chunks = (1, min(paintera_labels.shape[1], int(1e6)))
        f.create_dataset(out_key, data=paintera_labels, chunks=chunks,
                         compression='gzip')


def zero_out_ids(node_label_in_path, node_label_in_key,
                 node_label_out_path, node_label_out_key,
                 zero_ids):
    with open_file(node_label_in_path, 'r') as f:
        ds = f[node_label_in_key]
        node_labels = ds[:]
        chunks = ds.chunks

    zero_mask = np.isin(node_labels, zero_ids)
    node_labels[zero_mask] = 0

    with open_file(node_label_out_path) as f:
        ds = f.require_dataset(node_label_out_key, shape=node_labels.shape, chunks=chunks,
                               compression='gzip', dtype=node_labels.dtype)
        ds[:] = node_labels


# TODO refactor this properly
def get_bounding_boxes(table_path, table_key, scale_factor):
    with open_file(table_path, 'r') as f:
        table = f[table_key][:]
    bb_starts = table[:, 5:8]
    bb_stops = table[:, 8:]
    bb_starts /= scale_factor
    bb_stops /= scale_factor
    bounding_boxes = [tuple(slice(int(floor(sta)),
                                  int(ceil(sto))) for sta, sto in zip(start, stop))
                      for start, stop in zip(bb_starts, bb_stops)]
    return bounding_boxes


def check_exported(res_file, raw_path, raw_key, ws_path, ws_key,
                   table_path, table_key, scale_factor):
    from heimdall import view
    import nifty.tools as nt
    seg_id = int(os.path.splitext(os.path.split(res_file)[1])[0])
    res = np.load(res_file)

    bb = get_bounding_boxes(table_path, table_key, scale_factor)[seg_id]

    node_ids, node_labels = res['node_ids'], res['node_labels']
    assert len(node_ids) == len(node_labels)

    with open_file(raw_path, 'r') as f:
        ds = f[raw_key]
        ds.n_threads = 8
        raw = ds[bb]
    with open_file(ws_path, 'r') as f:
        ds = f[ws_key]
        ds.n_threads = 8
        ws = ds[bb]

    seg_mask = np.isin(ws, node_ids)
    ws[~seg_mask] = 0
    label_dict = {wsid: lid for wsid, lid in zip(node_ids, node_labels)}
    label_dict[0] = 0
    seg = nt.takeDict(label_dict, ws)

    view(raw, seg)


def check_exported_paintera(paintera_path, assignment_key,
                            node_label_path, node_label_key,
                            table_path, table_key, scale_factor,
                            raw_path, raw_key, seg_path, seg_key,
                            check_ids):
    from heimdall import view
    import nifty.tools as nt

    with open_file(paintera_path, 'r') as f:
        ds = f[assignment_key]
        new_assignments = ds[:].T

    with open_file(node_label_path, 'r') as f:
        ds = f[node_label_key]
        node_labels = ds[:]

    bounding_boxes = get_bounding_boxes(table_path, table_key, scale_factor)
    with open_file(seg_path, 'r') as fseg, open_file(raw_path, 'r') as fraw:
        ds_seg = fseg[seg_key]
        ds_seg.n_thread = 8
        ds_raw = fraw[raw_key]
        ds_raw.n_thread = 8

        for seg_id in check_ids:
            bb = bounding_boxes[seg_id]
            raw = ds_raw[bb]
            ws = ds_seg[bb]

            ws_ids = np.where(node_labels == seg_id)[0]
            seg_mask = np.isin(ws, ws_ids)
            ws[~seg_mask] = 0

            new_label_mask = np.isin(new_assignments[:, 0], ws_ids)
            new_label_dict = dict(zip(new_assignments[:, 0][new_label_mask],
                                      new_assignments[:, 1][new_label_mask]))
            new_label_dict[0] = 0

            # I am not sure why this happens
            un_ws = np.unique(ws)
            un_labels = list(new_label_dict.keys())
            missing = np.setdiff1d(un_ws, un_labels)
            print("Number of missing: ")
            new_label_dict.update({miss: 0 for miss in missing})

            seg_new = nt.takeDict(new_label_dict, ws)
            view(raw, seg_mask.astype('uint32'), seg_new)
