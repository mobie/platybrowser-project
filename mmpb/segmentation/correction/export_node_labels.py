import os
import json

from datetime import datetime
from glob import glob
from math import ceil, floor
from shutil import copytree, copy

import numpy as np
from elf.io import open_file
from elf.io.label_multiset_wrapper import LabelMultisetWrapper


def backup(path, key):
    file_path = os.path.join(path, key)
    timestamp = str(datetime.timestamp(datetime.now())).replace('.', '-')
    bkp_path = file_path + '.' + timestamp
    print("Make backup of", file_path, "in", bkp_path)
    copytree(file_path, bkp_path)


def backup_attrs(path):
    timestamp = str(datetime.timestamp(datetime.now())).replace('.', '-')
    bkp_path = path + '.' + timestamp
    print("Make backup of", path, "in", bkp_path)
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


def get_index_permutation(x, y):
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    return xsorted[ypos]


def export_node_labels(path, assignment_key, project_folder, id_offset):
    with open_file(path, 'r') as f:
        ds = f[assignment_key]
        node_labels = ds[:].T
    fragment_ids, node_labels = node_labels[:, 0], node_labels[:, 1]
    assert len(fragment_ids) == len(node_labels)

    result_folder = os.path.join(project_folder, 'splitting_tool', 'results')
    res_files = glob(os.path.join(result_folder, '*.npz'))
    print("Applying changes for", len(res_files), "resolved objects")

    resolved_ids = []
    for resf in res_files:
        seg_id = int(os.path.splitext(os.path.split(resf)[1])[0])
        print("Resolving", seg_id)
        res = np.load(resf)

        this_ids, this_labels = res['node_ids'], res['node_labels']
        assert len(this_ids) == len(this_labels)
        id_mask = node_labels == seg_id
        this_ids_exp = fragment_ids[id_mask]
        assert len(this_ids) == len(this_ids_exp), "%i, %i" % (len(this_ids), len(this_ids_exp))
        assert np.array_equal(np.sort(this_ids), np.sort(this_ids_exp))
        ids_sorted = get_index_permutation(this_ids, this_ids_exp)
        assert np.array_equal(this_ids[ids_sorted], this_ids_exp)

        this_labels += id_offset
        node_labels[id_mask] = this_labels[ids_sorted]
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


def get_bounding_boxes(table_path, table_key, scale_factor):
    with open_file(table_path, 'r') as f:
        ds = f[table_key]
        print(ds.shape)
        ds.n_threads = 8
        bb_starts = ds[:, 5:8]
        bb_stops = ds[:, 8:]
    bb_starts /= scale_factor
    bb_stops /= scale_factor
    bounding_boxes = [tuple(slice(int(floor(sta)),
                                  int(ceil(sto))) for sta, sto in zip(start, stop))
                      for start, stop in zip(bb_starts, bb_stops)]
    return bounding_boxes


def check_exported(paintera_path, old_assignment_key, assignment_key,
                   table_path, table_key, scale_factor,
                   raw_path, raw_key, ws_path, ws_key, check_ids):
    print("Start to check exported node labels")
    import napari
    import nifty.tools as nt

    with open_file(paintera_path, 'r') as f:
        ds = f[old_assignment_key]
        ds.n_threads = 8
        old_assignments = ds[:].T

        ds = f[assignment_key]
        ds.n_threads = 8
        assignments = ds[:].T

    fragment_ids, segment_ids = assignments[:, 0], assignments[:, 1]
    old_fragment_ids, old_segment_ids = old_assignments[:, 0], old_assignments[:, 1]
    assert np.array_equal(fragment_ids, old_fragment_ids)

    print("Loading bounding boxes ...")
    bounding_boxes = get_bounding_boxes(table_path, table_key, scale_factor)
    print("... done")
    with open_file(raw_path, 'r') as fraw, open_file(ws_path, 'r') as fws:

        ds_raw = fraw[raw_key]
        ds_raw.n_thread = 8

        ds_ws = fws[ws_key]
        ds_ws.n_thread = 8
        ds_ws = LabelMultisetWrapper(ds_ws)

        for seg_id in check_ids:
            print("Check object", seg_id)
            bb = bounding_boxes[seg_id]
            print("Within bounding box", bb)

            raw = ds_raw[bb]
            ws = ds_ws[bb]

            id_mask = old_segment_ids == seg_id
            ws_ids = fragment_ids[id_mask]
            seg_mask = np.isin(ws, ws_ids)
            ws[~seg_mask] = 0

            ids_old = old_segment_ids[id_mask]
            dict_old = {wid: oid for wid, oid in zip(ws_ids, ids_old)}
            dict_old[0] = 0
            seg_old = nt.takeDict(dict_old, ws)

            ids_new = segment_ids[id_mask]
            dict_new = {wid: oid for wid, oid in zip(ws_ids, ids_new)}
            dict_new[0] = 0
            seg_new = nt.takeDict(dict_new, ws)

            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_image(raw, name='raw')
                viewer.add_labels(seg_mask, name='seg-mask')
                viewer.add_labels(seg_old, name='old-seg')
                viewer.add_labels(seg_new, name='new-seg')
