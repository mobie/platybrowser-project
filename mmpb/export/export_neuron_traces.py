import os
from glob import glob

import numpy as np
import pandas as pd
import elf.skeleton.io as skio
from elf.io import open_file
from skimage.draw import circle
from pybdv.converter import make_scales
from pybdv.metadata import write_xml_metadata, write_h5_metadata, write_n5_metadata, get_data_path
from pybdv.util import get_key
from tqdm import tqdm
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


def write_vol_from_traces(traces, out_path, key, shape, resolution, chunks,
                          radius, n_threads, crop_overhanging=True):
    # write temporary h5 dataset
    # and write coordinates (with some radius) to it
    with open_file(out_path) as f:
        ds = f.require_dataset(key, shape=shape, dtype='int16', compression='gzip',
                               chunks=chunks)
        ds.n_threads = n_threads
        for nid, vals in tqdm(traces.items()):
            coords = vals_to_coords(vals, resolution)
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0) + 1
            assert all(bmi < bma for bmi, bma in zip(bb_min, bb_max))
            this_trace = coords_to_vol(coords, nid, radius=radius)

            if any(b > sh for b, sh in zip(bb_max, shape)):
                if crop_overhanging:
                    crop = [max(int(b - sh), 0) for b, sh in zip(bb_max, shape)]
                    print("Cropping by", crop)
                    vol_bb = tuple(slice(0, sh - cr)
                                   for sh, cr in zip(this_trace.shape, crop))
                    this_trace = this_trace[vol_bb]
                    bb_max = [b - crp for b, crp in zip(bb_max, crop)]
                else:
                    raise RuntimeError("Invalid bounding box: %s, %s" % (str(bb_max),
                                                                         str(shape)))

            bb = tuple(slice(int(bmi), int(bma)) for bmi, bma in zip(bb_min, bb_max))

            sub_vol = ds[bb]
            trace_mask = this_trace != 0
            sub_vol[trace_mask] = this_trace[trace_mask]
            ds[bb] = sub_vol


def traces_to_volume(traces, reference_vol_path, reference_scale, out_path,
                     resolution, scale_factors, radius=2, chunks=None, n_threads=8):
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
    print("Writing traces ...")
    write_vol_from_traces(traces, out_path, key0, shape, resolution, chunks, radius,
                          n_threads)

    print("Downscaling traces ...")
    make_scales(out_path, scale_factors, downscale_mode='max',
                ndim=3, setup_id=0, is_h5=is_h5,
                chunks=chunks, n_threads=n_threads)

    xml_path = os.path.splitext(out_path)[0] + '.xml'
    # we assume that the resolution is in nanometer, but want to write in microns for bdv
    bdv_res = [res / 1000. for res in resolution]
    unit = 'micrometer'
    write_xml_metadata(xml_path, out_path, unit, bdv_res, is_h5)
    bdv_scale_factors = [[1, 1, 1]] + scale_factors
    if is_h5:
        write_h5_metadata(out_path, bdv_scale_factors)
    else:
        write_n5_metadata(out_path, bdv_scale_factors, bdv_res)


def make_traces_table(traces, reference_scale, resolution, out_path, seg_infos={}):
    """ Make table from traces compatible with the platy browser.
    """

    files = {}
    datasets = {}
    for seg_name, seg_info in seg_infos.items():

        seg_path = seg_info['path']
        if seg_path.endswith('.xml'):
            seg_path = get_data_path(seg_path, return_absolute_path=True)
        seg_scale = seg_info['scale']
        is_h5 = is_h5_file(seg_path)
        seg_key = get_key(is_h5, time_point=0, setup_id=0, scale=seg_scale)
        f = open_file(seg_path, 'r')
        ds = f[seg_key]

        if len(files) == 0:
            ref_shape = ds.shape
        else:
            assert ds.shape == ref_shape, "%s, %s" % (str(ds.shape), str(ref_shape))

        files[seg_name] = f
        datasets[seg_name] = ds

    table = []
    for nid, vals in tqdm(traces.items()):

        coords = vals_to_coords(vals, resolution)
        bb_min = coords.min(axis=0)
        bb_max = coords.max(axis=0) + 1

        # get spatial attributes
        anchor = coords[0].astype('float32') * resolution / 1000.
        bb_min = bb_min.astype('float32') * resolution / 1000.
        bb_max = bb_max.astype('float32') * resolution / 1000.

        # get cell and nucleus ids
        point_slice = tuple(slice(int(c), int(c) + 1) for c in coords[0])
        # attributes:
        # label_id
        # anchor_x anchor_y anchor_z
        # bb_min_x bb_min_y bb_min_z bb_max_x bb_max_y bb_max_z
        # n_points + seg ids
        attributes = [nid, anchor[2], anchor[1], anchor[0],
                      bb_min[2], bb_min[1], bb_min[0],
                      bb_max[2], bb_max[1], bb_max[0],
                      len(coords)]

        for ds in datasets.values():
            seg_id = ds[point_slice][0, 0, 0]
            attributes += [seg_id]

        table.append(attributes)

    for f in files.values():
        f.close()

    table = np.array(table, dtype='float32')
    header = ['label_id', 'anchor_x', 'anchor_y', 'anchor_z',
              'bb_min_x', 'bb_min_y', 'bb_min_z',
              'bb_max_x', 'bb_max_y', 'bb_max_z',
              'n_points']
    header += ['%s_id' % seg_name for seg_name in seg_infos]

    table = pd.DataFrame(table, columns=header)
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
