import csv
import os
from glob import glob

import numpy as np
import h5py
import elf.skeleton.io as skio
from skimage.draw import circle
from pybdv import convert_to_bdv


def extract_neuron_traces(trace_folder, reference_vol_path,
                          seg_out_path, table_out_path, tmp_folder,
                          cell_seg_info, nucleus_seg_info,
                          reference_scale=3):
    """ Extract all traced neurons stored in nmx format and export them
    as segmentation and table compatible with the platy browser.
    """
    os.makedirs(tmp_folder, exist_ok=True)
    trace_files = glob(os.path.join(trace_folder, "*.nmx"))
    # load all traces
    print("Load traces")
    traces = extract_traces(trace_files)
    if not traces:
        raise RuntimeError("Did not find any traces in %s" % trace_folder)
    print("Found", len(traces), "traces")

    # check that we are compatible with bdv (ids smaller )
    max_id = np.iinfo('int16').max
    max_trace_id = max(traces.keys())
    if max_trace_id > max_id:
        raise RuntimeError("Can't export id %i > %i" % (max_trace_id, max_id))

    # make table
    print("Make table")
    table, col_names = make_table(traces, reference_scale, cell_seg_info, nucleus_seg_info)
    write_table(table, col_names, table_out_path)
    return

    # make segmentation in tmp location and compy it to the output path
    print("Make segmentation")
    seg_tmp = os.path.join(tmp_folder, "traces_seg.h5")
    make_seg(traces, reference_vol_path, reference_scale, seg_tmp)
    traces_to_bdv(seg_tmp, seg_out_path, reference_scale)


def extract_traces(files):
    coords = {}
    for path in files:
        skel = skio.read_nml(path)
        search_str = 'neuron_id'
        for k, v in skel.items():
            # for now, we only extract nodes belonging to
            # what's annotated as 'skeleton'. There are also tags for
            # 'soma' and 'synapse'. I am ignoring these for now.

            # is_soma = 'soma' in k
            # is_synapse = 'synapse' in k
            is_skeleton = 'skeleton' in k
            if not is_skeleton:
                continue

            sub = k.find(search_str)
            beg = sub + len(search_str)
            end = k.find('.', beg)
            n_id = int(k[beg:end])

            # make sure we keep the order of keys when extracting the
            # values
            kvs = v.keys()
            c = [vv for kv in sorted(kvs) for vv in v[kv]]
            if n_id in coords:
                coords[n_id].extend(c)
            else:
                coords[n_id] = c
    return coords


def get_resolution(scale, use_nm=True):
    if use_nm:
        res0 = [25, 10, 10]
        res1 = [25, 20, 20]
    else:
        res0 = [0.025, 0.01, 0.01]
        res1 = [0.025, 0.02, 0.02]
    resolutions = [res0] + [[re * (2 ** (i)) for re in res1] for i in range(5)]
    return np.array(resolutions[scale])


def coords_to_vol(coords, nid, radius=5):
    bb_min = coords.min(axis=0)
    bb_max = coords.max(axis=0) + 1

    sub_shape = tuple(bma - bmi for bmi, bma in zip(bb_min, bb_max))
    sub_vol = np.zeros(sub_shape, dtype='int16')
    sub_coords = coords - bb_min

    xy_shape = sub_vol.shape[1:]
    for c in sub_coords:
        z, y, x = c
        mask = circle(y, x, radius, shape=xy_shape)
        sub_vol[z][mask] = nid

    return sub_vol


def write_table(data, col_names, output_path):
    assert data.shape[1] == len(col_names), "%i %i" % (data.shape[1],
                                                       len(col_names))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(col_names)
        writer.writerows(data)


def vals_to_coords(vals, res):
    coords = np.array(vals)
    coords /= res
    coords = coords.astype('uint64')
    return coords


def make_table(traces, reference_scale, cell_seg_info, nucleus_seg_info):

    res = get_resolution(reference_scale)

    cell_path = cell_seg_info['path']
    cell_scale = cell_seg_info['scale']
    cell_key = 't00000/s00/%i/cells' % cell_scale

    nucleus_path = nucleus_seg_info['path']
    nucleus_scale = nucleus_seg_info['scale']
    nucleus_key = 't00000/s00/%i/cells' % nucleus_scale

    table = []
    with h5py.File(cell_path, 'r') as fc, h5py.File(nucleus_path, 'r') as fn:
        dsc = fc[cell_key]
        dsn = fn[nucleus_key]
        assert dsc.shape == dsn.shape, "%s, %s" % (str(dsc.shape), str(dsn.shape))

        for nid, vals in traces.items():
            coords = vals_to_coords(vals, res)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1

            # get spatial attributes
            anchor = coords[0].astype('float32') * res / 1000.
            bb_min = bb_min.astype('float32') * res / 1000.
            bb_max = bb_max.astype('float32') * res / 1000.

            # get cell and nucleus ids
            point_slice = tuple(slice(int(c), int(c) + 1) for c in coords[0])
            cell_id = dsc[point_slice][0, 0, 0]
            nucleus_id = dsn[point_slice][0, 0, 0]

            # attributes:
            # label_id
            # anchor_x anchor_y anchor_z
            # bb_min_x bb_min_y bb_min_z bb_max_x bb_max_y bb_max_z
            # n_points cell-id nucleus-id
            attributes = [nid, anchor[2], anchor[1], anchor[0],
                          bb_min[2], bb_min[1], bb_min[0],
                          bb_max[2], bb_max[1], bb_max[0],
                          len(coords), cell_id, nucleus_id]
            table.append(attributes)

    table = np.array(table, dtype='float32')
    header = ['label_id', 'anchor_x', 'anchor_y', 'anchor_z',
              'bb_min_x', 'bb_min_y', 'bb_min_z',
              'bb_max_x', 'bb_max_y', 'bb_max_z',
              'n_points', 'cell_id', 'nucleus_id']
    return table, header


def make_seg(traces, reference_vol_path, reference_scale, seg_out_path):

    # I assume that the coordinates have a resoultion of 1x1x1 nm
    # also, coords are in axis order x, y, z
    ref_key = 't00000/s00/%i/cells' % reference_scale
    with h5py.File(reference_vol_path, 'r') as f:
        shape = f[ref_key].shape
    res = get_resolution(reference_scale)

    # the circle radius we write out
    radius = 5

    # write temporary h5 dataset
    # and write coordinates (with some radius) to it
    with h5py.File(seg_out_path) as f:
        ds = f.require_dataset('traces', shape=shape, dtype='int16', compression='gzip')
        for nid, vals in traces.items():
            coords = vals_to_coords(vals, res)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1
            assert all(bmi < bma for bmi, bma in zip(bb_min, bb_max))
            assert all(b < sh for b, sh in zip(bb_max, shape))

            sub_vol = coords_to_vol(coords, nid, radius=radius)
            bb = tuple(slice(bmi, bma) for bmi, bma in zip(bb_min, bb_max))
            ds[bb] += sub_vol


# we could replace this with cluster_tools functionality if this becomes a bottlenecl
def traces_to_bdv(in_path, out_path, reference_scale):
    key = 'traces'
    scale_factors = [2, 2, 2, 2, 2]
    res = get_resolution(reference_scale, use_nm=False)
    convert_to_bdv(in_path, key, out_path,
                   resolution=res, unit='micrometer',
                   downscale_factors=scale_factors,
                   downscale_mode='max')
