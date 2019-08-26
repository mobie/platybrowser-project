import csv
import os
from glob import glob

import numpy as np
import h5py
import elf.skeletion.io as skio
from skimage.draw import circle
from pybdv import convert_to_bdv


def extract_neuron_traces(trace_folder, reference_vol_path,
                          seg_out_path, table_out_path, tmp_folder,
                          reference_scale=3):
    """ Extract all traced neurons stored in nmx format and export them
    as segmentation and table compatible with the platy browser.
    """
    os.makedirs(tmp_folder, exist_ok=True)
    trace_files = glob(os.path.join(trace_folder, "*.nmx"))
    # load all traces
    traces = extract_traces(trace_files)
    if not traces:
        raise RuntimeError("Did not find any traces in %s" % trace_folder)

    # make segmentation in tmp location and get table
    seg_tmp = os.path.join(tmp_folder, "traces_seg.h5")
    table, col_names = make_seg_and_scale(traces, reference_vol_path, reference_scale, seg_tmp)

    # copy segmentation to the output path and write table
    traces_to_bdv(seg_tmp, seg_out_path, reference_scale)
    write_table(table, col_names, table_out_path)


def extract_traces(files):
    coords = {}
    for path in files:
        skel = skio.read_nml(path)
        search_str = 'neuron_id'
        for k, v in skel.items():
            sub = k.find(search_str)
            beg = sub + len(search_str)
            end = k.find('.', beg)
            n_id = int(k[beg:end])
            c = [vv for vals in v.values() for vv in vals]
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
    return resolutions[scale]


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


def make_seg_and_scale(traces, reference_vol_path, reference_scale, seg_out_path):
    # I assume that the coordinates have a resoultion of 1x1x1 nm
    # also, coords are in axis order x, y, z

    ref_key = 't00000/s00/%i/cells' % reference_scale
    with h5py.File(reference_vol_path, 'r') as f:
        shape = f[ref_key].shape
    res = get_resolution(reference_scale)

    # the circle radius we write out
    radius = 10

    max_id = np.iinfo('int16').max

    # write temporary h5 dataset
    # and write coordinates (with some radius) to it
    table = []
    with h5py.File(seg_out_path) as f:
        ds = f.require_dataset('traces', shape=shape, dtype='int16', compression='gzip')
        for nid, vals in traces.items():
            print("Neuron id:", nid)
            if nid > max_id:
                raise RuntimeError("Can't export id %i > %i" % (nid, max_id))
            coords = np.array(vals)
            coords = coords[::-1]
            coords /= np.array(res)
            coords = coords.astype('uint64')

            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1
            assert all(bmi < bma for bmi, bma in zip(bb_min, bb_max))
            assert all(b < sh for b, sh in zip(bb_max, shape))

            sub_vol = coords_to_vol(coords, nid, radius=radius)
            bb = tuple(slice(bmi, bma) for bmi, bma in zip(bb_min, bb_max))
            ds[bb] += sub_vol

            # TODO we want the anchor to correspond to node0. I don't know if this
            # is currently extracted correctly in the coordinates
            # attributes:
            # label_id anchor_x anchor_y anchor_z bb_min_x bb_min_y bb_min_z bb_max_x bb_max_y bb_max_z n_points
            anchor = coords[0].astype('float32') * np.array(res) / 1000.
            bb_min = bb_min.astype('float32') * np.array(res) / 1000.
            bb_max = bb_max.astype('float32') * np.array(res) / 1000.
            attributes = [nid, anchor[2], anchor[1], anchor[0],
                          bb_min[2], bb_min[1], bb_min[0],
                          bb_max[2], bb_max[1], bb_max[0],
                          len(coords)]
            table.append(attributes)

    table = np.array(table, dtype='float32')
    print(table.shape)
    print(table.dtype)
    header = ['label_id', 'anchor_x', 'anchor_y', 'anchor_z',
              'bb_min_x', 'bb_min_y', 'bb_min_z',
              'bb_max_x', 'bb_max_y', 'bb_max_z',
              'n_points']
    return table, header


# we could replace this with cluster_tools functionality if this becomes a bottlenecl
def traces_to_bdv(in_path, out_path, reference_scale):
    key = 'traces'
    scale_factors = [2, 2, 2, 2, 2]
    res = get_resolution(reference_scale, use_nm=False)
    convert_to_bdv(in_path, key, out_path,
                   resolution=res, unit='micrometer',
                   downscale_factors=scale_factors,
                   downscale_mode='max')
