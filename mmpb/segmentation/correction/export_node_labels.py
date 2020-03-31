import os
import json
from datetime import datetime
from glob import glob
from math import ceil, floor
from shutil import copytree, copy

import numpy as np
from elf.io import open_file


def backup(path, key):
    file_path = os.path.join(path, key)
    timestamp = str(datetime.timestamp(datetime.now())).replace('.', '-')
    bkp_path = file_path + '.' + timestamp
    copytree(file_path, bkp_path)


def backup_attrs(path):
    timestamp = str(datetime.timestamp(datetime.now())).replace('.', '-')
    bkp_path = path + '.' + timestamp
    copy(path, bkp_path)


def read_paintera_max_id(project_folder):
    path = os.path.join(project_folder, 'data.n5')
    with open_file(path, 'r') as f:
        ds = f['volumes/paintera']
        max_id = ds.attrs['maxId']
    return max_id


def write_paintera_max_id(project_folder, max_id):
    path = os.path.join(project_folder, 'data.n5')
    key = 'volumes/paintera'
    attrs_file = os.path.join(path, key, 'attributes.json')
    backup_attrs(attrs_file)
    with open_file(path) as f:
        ds = f[key]
        ds.attrs['maxId'] = max_id


def remove_flagged_ids(paintera_attrs, ids):
    backup_attrs(paintera_attrs)
    with open(paintera_attrs) as f:
        attrs = json.load(f)

    source_name = 'org.janelia.saalfeldlab.paintera.state.label.ConnectomicsLabelState'

    sources = attrs['paintera']['sourceInfo']['sources']
    have_seg_source = False
    for ii, source in enumerate(sources):
        type_ = source['type']
        if type_ == source_name:
            assert not have_seg_source, "Only support a single segmentation source!"
            source_state = source['state']

            flagged_ids = set(source_state['flaggedSegments'])
            flagged_ids = list(flagged_ids - set(ids))

            source_state['flaggedSegments'] = flagged_ids
            source['state'] = source_state
            sources[ii] = source

            have_seg_source = True

    assert have_seg_source, "Did not find any segmentation source"
    attrs['paintera']['sourceInfo']['sources'] = sources

    with open(paintera_attrs, 'w') as f:
        json.dump(attrs, f)


def export_node_labels(path, assignment_key, project_folder, id_offset):
    with open_file(path, 'r') as f:
        ds = f[assignment_key]
        node_labels = ds[:].T
    fragment_ids, node_labels = node_labels[:, 0], node_labels[:, 1]

    result_folder = os.path.join(project_folder, 'splitting_tool', 'results')
    res_files = glob(os.path.join(result_folder, '*.npz'))
    print("Applying changes for", len(res_files), "resolved objects")

    resolved_ids = []
    for resf in res_files:
        seg_id = int(os.path.splitext(os.path.split(resf)[1])[0])
        res = np.load(resf)

        this_ids, this_labels = res['node_ids'], res['node_labels']
        assert len(this_ids) == len(this_labels)
        id_mask = node_labels == seg_id
        this_ids_exp = fragment_ids[id_mask]
        assert np.array_equal(np.sort(this_ids), np.sort(this_ids_exp))

        this_labels += id_offset
        node_labels[id_mask] = this_labels
        id_offset = int(this_labels.max()) + 1

        resolved_ids.append(seg_id)

    backup(path, assignment_key)

    paintera_labels = np.concatenate([fragment_ids[:, None], node_labels[:, None]], axis=1).T
    with open_file(path) as f:
        ds = f[assignment_key]
        ds[:] = paintera_labels

    return id_offset, resolved_ids


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
