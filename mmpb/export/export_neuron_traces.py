import os
from glob import glob

import numpy as np
import pandas as pd
import elf.skeleton.io as skio
from elf.io import open_file
from skimage.draw import circle
from pybdv.converter import make_scales
from pybdv.metadata import write_xml_metadata, write_h5_metadata, write_n5_metadata
from pybdv.util import get_key
from ..util import is_h5_file


def extract_neuron_traces_from_nmx(trace_folder):
    """Extract all traced neurons stored in nmx format and return as dict.
    """
    trace_files = glob(os.path.join(trace_folder, "*.nmx"))
    if not trace_files:
        raise RuntimeError("Did not find any traces in %s" % trace_folder)
    coords = {}
    for path in trace_files:
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


def write_vol_from_traces(traces, out_path, key, shape, resolution, chunks, radius):
    # write temporary h5 dataset
    # and write coordinates (with some radius) to it
    with open_file(out_path) as f:
        ds = f.require_dataset(key, shape=shape, dtype='int16', compression='gzip',
                               chunks=chunks)
        for nid, vals in traces.items():
            coords = vals_to_coords(vals, resolution)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1
            assert all(bmi < bma for bmi, bma in zip(bb_min, bb_max))
            assert all(b < sh for b, sh in zip(bb_max, shape))

            sub_vol = coords_to_vol(coords, nid, radius=radius)
            bb = tuple(slice(bmi, bma) for bmi, bma in zip(bb_min, bb_max))
            ds[bb] += sub_vol


# TODO this takes ages, somehow parallelize ! (need to lock somehow though)
def traces_to_volume(traces, reference_vol_path, reference_scale, out_path,
                     resolution, scale_factors, radius=5, chunks=None, n_threads=8):
    """ Export traces as segmentation compatible with the platy-browser.
    """

    # check that we are compatible with bdv (ids need to be smaller than int16 max)
    max_id = np.iinfo('int16').max
    max_trace_id = max(traces.keys())
    if max_trace_id > max_id:
        raise RuntimeError("Can't export id %i > %i" % (max_trace_id, max_id))

    is_h5 = is_h5_file(reference_vol_path)
    ref_key = get_key(is_h5, time_point=0, setup_id=0, scale=reference_scale)
    with open_file(reference_vol_path, 'r') as f:
        ds = f[ref_key]
        shape = ds.shape
        if chunks is None:
            chunks = ds.chunks

    is_h5 = is_h5_file(out_path)
    key0 = get_key(is_h5, time_point=0, setup_id=0, scale=0)
    write_vol_from_traces(traces, out_path, key0, shape, resolution, chunks, radius)

    make_scales(out_path, scale_factors, downscale_mode='max',
                ndim=3, setup_id=0, is_h5=is_h5,
                chunks=chunks, n_threads=n_threads)

    xml_path = os.path.splitext(out_path)[0] + '.xml'
    # we assume that the resolution is in nanometer, but want to write in microns for bdv
    bdv_res = [res / 1000. for res in resolution]
    unit = 'micrometer'
    write_xml_metadata(xml_path, out_path, unit, bdv_res, is_h5)
    if is_h5:
        write_h5_metadata(out_path, scale_factors)
    else:
        write_n5_metadata(out_path, scale_factors, bdv_res)


# FIXME currently not working properly
# TODO parallelize
# TODO support passing a dict of seg-infos instead of hard-coding it to cell and nucleus seg
def make_traces_table(traces, reference_scale, resolution, out_path,
                      cell_seg_info, nucleus_seg_info):
    """ Make table from traces compatible with the platy browser.
    """

    cell_path = cell_seg_info['path']
    cell_scale = cell_seg_info['scale']
    is_h5 = is_h5_file(cell_path)
    cell_key = get_key(is_h5, time_point=0, setup_id=0, scale=cell_scale)

    nucleus_path = nucleus_seg_info['path']
    nucleus_scale = nucleus_seg_info['scale']
    is_h5 = is_h5_file(nucleus_path)
    nucleus_key = get_key(is_h5, time_point=0, setup_id=0, scale=nucleus_scale)

    table = []
    with open_file(cell_path, 'r') as fc, open_file(nucleus_path, 'r') as fn:
        dsc = fc[cell_key]
        dsn = fn[nucleus_key]
        assert dsc.shape == dsn.shape, "%s, %s" % (str(dsc.shape), str(dsn.shape))

        for nid, vals in traces.items():
            coords = vals_to_coords(vals, resolution)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1

            # get spatial attributes
            anchor = coords[0].astype('float32') * resolution / 1000.
            bb_min = bb_min.astype('float32') * resolution / 1000.
            bb_max = bb_max.astype('float32') * resolution / 1000.

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

    table = pd.DataFrame(table, header)
    table.to_csv(out_path, index=False, sep='\t')


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


def vals_to_coords(vals, res):
    coords = np.array(vals)
    coords /= res
    coords = coords.astype('uint64')
    return coords
