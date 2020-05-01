import os
import numpy as np
import pandas as pd
import z5py

from elf.mesh import marching_cubes
from elf.mesh.io import write_obj
from pybdv.metadata import get_data_path

ROOT = '../../data'


def gene_to_ids(version, gene_name):
    # get the normal table to check which ones are actually cells
    table = os.path.join(ROOT, version, 'tables',
                         'sbem-6dpf-1-whole-segmented-cells', 'default.csv')
    table = pd.read_csv(table, sep='\t')
    cell_ids = table['label_id'].values.astype('uint32')
    is_cell = table['cells'].values.astype('bool')

    # get the cells assigned to our gene
    gene_table = os.path.join(ROOT, version, 'tables',
                              'sbem-6dpf-1-whole-segmented-cells', 'vc_assignments.csv')
    gene_table = pd.read_csv(gene_table, sep='\t')
    gene_column = gene_table[gene_name].values.astype('bool')

    assert len(gene_column) == len(is_cell) == len(cell_ids)
    expressed = np.logical_and(is_cell, gene_column)

    return cell_ids[expressed]


def get_resolution(scale):
    r0 = [0.025, 0.02, 0.02]
    res = [[rr * 2 ** ii for rr in r0] for ii in range(10)]
    return res[scale]


def load_bounding_boxes(table_path, resolution):
    table = pd.read_csv(table_path, sep='\t')
    bb_start = table[['bb_min_z', 'bb_min_y', 'bb_min_x']].values.astype('float')
    bb_stop = table[['bb_max_z', 'bb_max_y', 'bb_max_x']].values.astype('float')
    bb_start /= np.array(resolution)
    bb_stop /= np.array(resolution)
    return bb_start, bb_stop


def export_mesh(label_id, ds, bb_starts, bb_stops, resolution, out_path):
    start, stop = bb_starts[label_id], bb_stops[label_id]
    bb = tuple(slice(int(sta), int(sto) + 1) for sta, sto in zip(start, stop))

    seg = ds[bb]
    mask = seg == label_id
    n_foreground = mask.sum()
    # assert we have some meaningful number of foreground pixels
    assert n_foreground > 100

    res_nm = [1000. * re for re in resolution]
    verts, faces, normals = marching_cubes(mask, resolution=res_nm)
    # go from zyx axis convention to xyz
    verts = verts[:, ::-1]
    normals = normals[:, ::-1]

    # offset the vertex coordinates
    offset = np.array([sta * re for sta, re in zip(start, res_nm)])[::-1]
    verts += offset

    # save to obj
    write_obj(out_path, verts, faces, normals)


def gene_to_meshes(version, gene_name, out_folder, scale=2):
    os.makedirs(out_folder, exist_ok=True)
    cell_ids = gene_to_ids(version, gene_name)

    # load the segmentation dataset
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-cells.xml')
    path = get_data_path(xml_path, return_absolute_path=True)
    key = 'setup0/timepoint0/s%i' % scale
    f = z5py.File(path, 'r')
    ds = f[key]
    ds.n_threads = 8

    # load the default table to get the bounding boxes
    table_path = os.path.join(ROOT, version, 'tables/sbem-6dpf-1-whole-segmented-cells/default.csv')
    resolution = get_resolution(scale)
    bb_starts, bb_stops = load_bounding_boxes(table_path, resolution)

    # TODO do for all cell ids in parallel
    print("Computing mesh ...")
    out_path = os.path.join(out_folder, 'mesh_%i.obj' % cell_ids[0])
    export_mesh(cell_ids[0], ds, bb_starts, bb_stops, resolution, out_path)


if __name__ == '__main__':
    version = '1.0.0'
    name = 'phc2'
    out_folder = './meshes_phc2'
    gene_to_meshes(version, name, out_folder)
