import argparse
import os

import numpy as np
import pandas as pd
from mmpb.export.meshes import export_meshes

ROOT = '../../data'


def get_muscle_ids(version, n_meshes):
    # get the normal table to check which ones are actually cells
    table = os.path.join(ROOT, version, 'tables',
                         'sbem-6dpf-1-whole-segmented-cells', 'default.csv')
    table = pd.read_csv(table, sep='\t')
    cell_ids = table['label_id'].values.astype('uint32')
    is_cell = table['cells'].values.astype('bool')

    # get the cells which are muscles
    muscle_table = os.path.join(ROOT, version, 'tables',
                                'sbem-6dpf-1-whole-segmented-cells', 'regions.csv')
    muscle_table = pd.read_csv(muscle_table, sep='\t')
    muscle_column = muscle_table['muscle'].values.astype('bool')

    assert len(muscle_column) == len(is_cell) == len(cell_ids)
    is_muscle = np.logical_and(is_cell, muscle_column)

    return cell_ids[is_muscle][:n_meshes]


def get_resolution(scale):
    r0 = [0.025, 0.02, 0.02]
    res = [[rr * 2 ** ii for rr in r0] for ii in range(10)]
    return res[scale]


def muscle_meshes(version, n_meshes, out_folder, scale=3, n_jobs=16):
    cell_ids = get_muscle_ids(version, n_meshes)

    # load the segmentation dataset
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-cells.xml')
    table_path = os.path.join(ROOT, version, 'tables/sbem-6dpf-1-whole-segmented-cells/default.csv')
    resolution = get_resolution(scale)

    export_meshes(xml_path, table_path, cell_ids, out_folder, scale, resolution, n_jobs=16)


# TODO generalize this for other things from the region table
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export meshes for muscle cells")
    parser.add_argument('--n_meshes', type=str, default=16)
    parser.add_argument('--version', type=str, default='1.0.1')
    args = parser.parse_args()

    n_meshes = args.n_meshes
    version = args.version
    out_folder = './meshes_muscles'
    muscle_meshes(version, n_meshes, out_folder)
