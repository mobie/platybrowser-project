import os
from concurrent import futures

import numpy as np
import pandas as pd
import z5py
from elf.mesh import marching_cubes
from elf.mesh.io import write_obj
from elf.wrapper.resized_volume import ResizedVolume
from pybdv.metadata import get_data_path, get_resolution
from tqdm import tqdm


def load_bounding_boxes(table_path, resolution):
    table = pd.read_csv(table_path, sep='\t')
    bb_start = table[['bb_min_z', 'bb_min_y', 'bb_min_x']].values.astype('float')
    bb_stop = table[['bb_max_z', 'bb_max_y', 'bb_max_x']].values.astype('float')
    bb_start /= np.array(resolution)
    bb_stop /= np.array(resolution)
    return bb_start, bb_stop


def export_mesh(label_id, ds, bb_starts, bb_stops, resolution, out_path, ds_mask=None):
    if bb_starts is None:
        bb = np.s_[:]
    else:
        start, stop = bb_starts[label_id], bb_stops[label_id]
        bb = tuple(slice(int(sta), int(sto) + 1) for sta, sto in zip(start, stop))

    obj = (ds[bb] == label_id)

    if ds_mask is not None:
        mask = ds_mask[bb].astype('bool')
        obj[~mask] = 0

    n_foreground = obj.sum()
    # assert we have some meaningful number of foreground pixels
    assert n_foreground > 100

    res_nm = [1000. * re for re in resolution]
    verts, faces, normals = marching_cubes(obj, resolution=res_nm)
    # go from zyx axis convention to xyz
    verts = verts[:, ::-1]
    normals = normals[:, ::-1]

    # offset the vertex coordinates
    if bb_starts is not None:
        offset = np.array([sta * re for sta, re in zip(start, res_nm)])[::-1]
        verts += offset

    # save to obj
    write_obj(out_path, verts, faces, normals)


def export_meshes(xml_path, table_path, cell_ids, out_folder, scale, resolution=None, n_jobs=16,
                  xml_mask_path=None, mask_scale=None):
    os.makedirs(out_folder, exist_ok=True)

    if resolution is None:
        resolution = get_resolution(xml_path, 0)
        if scale > 0:
            resolution = [re * 2 ** scale for re in resolution]

    # load the segmentation dataset
    path = get_data_path(xml_path, return_absolute_path=True)
    key = 'setup0/timepoint0/s%i' % scale
    f = z5py.File(path, 'r')
    ds = f[key]
    ds.n_threads = 8

    # load the default table to get the bounding boxes
    if table_path is None:
        bb_starts, bb_stops = None, None
    else:
        bb_starts, bb_stops = load_bounding_boxes(table_path, resolution)

    if xml_mask_path is None:
        ds_mask = None
    else:
        mask_data_path = get_data_path(xml_mask_path, return_absolute_path=True)
        mask_key = 'setup0/timepoint0/s%i' % mask_scale
        f_mask = z5py.File(mask_data_path, 'r')
        ds_mask = f[mask_key]
        if ds_mask.shape != ds.shape:
            ds_mask = ResizedVolume(ds_mask, ds.shape, order=0)

    def _mesh(cell_id):
        out_path = os.path.join(out_folder, 'mesh_%i.obj' % cell_id)
        export_mesh(cell_id, ds, bb_starts, bb_stops, resolution, out_path, ds_mask)

    print("Computing meshes ...")
    with futures.ThreadPoolExecutor(n_jobs) as tp:
        list(tqdm(tp.map(_mesh, cell_ids), total=len(cell_ids)))
