import os
import numpy as np
import pandas as pd


# TODO need better cell nucleus mapping
def cell_counts(table_folder):
    """ Count the number of cells.

    Compute the following counts:
        - total number of cells
        - cells that have overlap with specific regions
        - muscle cells
    """

    # load the  nuclei mapping table
    nuclei_table = os.path.join(table_folder, 'cells_to_nuclei.csv')
    nuclei_table = pd.read_csv(nuclei_table, sep='\t')

    # filter for the cell ids that corresponds to actual cells
    # based on the nucleus mapping
    cell_mask = nuclei_table['nucleus_id'].values != 0
    n_cells = cell_mask.sum()

    # load the region / tissue / semantic table
    region_table = os.path.join(table_folder, 'regions.csv')
    region_table = pd.read_csv(region_table, sep='\t')
    assert region_table.shape[0] == cell_mask.shape[0]

    # count the number of cells for individual regions/muscle
    count_dict = {}
    names = ['gut', 'muscle', 'crypticsegment', 'pns', 'head', 'pygidium',
             'stomodeum', 'vnc']
    for col_name in region_table.columns:
        if col_name in names:
            col = region_table[col_name].values
            col_mask = col != 0
            col_sum = np.logical_and(cell_mask, col_mask).sum()
            count_dict[col_name] = col_sum

    return n_cells, count_dict


if __name__ == '__main__':
    n_cells, count_dict = cell_counts('../../data/0.1.1/tables/sbem-6dpf-1-whole-segmented-cells-labels')
    print("Number of cells:", n_cells)
    for name, count in count_dict.items():
        print("Number", name, "cells:", count)
