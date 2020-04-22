import os
import numpy as np
import pandas as pd

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


# we use the nuclei coordiantes
def ids_to_coordinates(version, cell_ids):
    nuc_table = os.path.join(ROOT, version, 'tables',
                             'sbem-6dpf-1-whole-segmented-cells', 'cells_to_nuclei.csv')
    nuc_table = pd.read_csv(nuc_table, sep='\t')

    label_ids = nuc_table['label_id'].values.astype('uint32')
    nuc_ids = nuc_table['nucleus_id'].values.astype('uint32')
    this_nuc_ids = nuc_ids[np.isin(label_ids, cell_ids)]

    nuc_centers = os.path.join(ROOT, version, 'tables',
                               'sbem-6dpf-1-whole-segmented-nuclei', 'default.csv')
    nuc_centers = pd.read_csv(nuc_centers, sep='\t')
    all_nuc_ids = nuc_centers['label_id'].astype('uint32')

    # TODO we need to bring this in the same coordinate convention as knossos.
    # I think it is x,y,z and coordinates are expected in nanometer
    nuc_centers = nuc_centers[['anchor_x', 'anchor_y', 'anchor_z']].values.astype('float32')
    nuc_centers *= 1000.
    this_mask = np.isin(all_nuc_ids, this_nuc_ids)

    this_centers = nuc_centers[this_mask]

    return this_centers


def gene_to_traces(version, gene_name, out):
    cell_ids = gene_to_ids(version, gene_name)
    print("The gene", gene_name, "is expressed in", len(cell_ids), "cells")
    center_coordinates = ids_to_coordinates(version, cell_ids)
    assert len(center_coordinates) == len(cell_ids)
    # TODO write the cell-ids and coordinates to nmx or other skeleton representation supported by knossos


if __name__ == '__main__':
    version = '1.0.0'
    name = 'phc2'
    out = './traces_phc2.nmx'
    gene_to_traces(version, name, out)
