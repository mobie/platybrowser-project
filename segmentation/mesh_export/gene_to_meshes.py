import argparse
import os

import numpy as np
import pandas as pd
from mmpb.export.meshes import export_meshes

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


def gene_to_meshes(version, gene_name, out_folder, scale=2, n_jobs=16):
    cell_ids = gene_to_ids(version, gene_name)

    # load the segmentation dataset
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-cells.xml')
    table_path = os.path.join(ROOT, version, 'tables/sbem-6dpf-1-whole-segmented-cells/default.csv')
    resolution = get_resolution(scale)

    export_meshes(xml_path, table_path, cell_ids, out_folder, scale, resolution, n_jobs=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export meshes for a specific gene (default: PhC2)")
    parser.add_argument('--gene', type=str, default='phc2')
    parser.add_argument('--version', type=str, default='1.0.0')
    args = parser.parse_args()

    name = args.gene
    version = args.version
    out_folder = f'./meshes_{name}'
    gene_to_meshes(version, name, out_folder)
