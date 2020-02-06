import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from concurrent import futures


def get_bb(idd, table, res):
    row = table.loc[idd]
    bb_min = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
    bb_max = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
    return tuple(slice(int(mi / re), int(ma / re))
                 for mi, ma, re in zip(bb_min, bb_max, res))


def match_cilia_to_cells(cell_ids, cell_table,
                         cilia_seg_path, cilia_seg_key, cilia_res):
    cell_table = pd.read_csv(cell_table, sep='\t')
    cell_table.set_index('label_id')

    with h5py.File(cilia_seg_path, 'r') as f:
        ds_cil = f[cilia_seg_key]

        def match_single_cell(cell_id):
            # get the bounding box of this cell
            bb = get_bb(cell_id, cell_table, cilia_res)
            seg = ds_cil[bb]
            return np.unique(seg)

        with futures.ThreadPoolExecutor(9) as tp:
            tasks = [tp.submit(match_single_cell, cell_id) for cell_id in cell_ids]
            res = [t.result() for t in tasks]

        cilia_ids = np.concatenate(res)
        cilia_ids = np.unique(cilia_ids)
    return cilia_ids


def plot_sizes(table):
    sizes = table['n_pixels'].values[1:]
    print(sizes.max(), sizes.min())

    fig, ax = plt.subplots()
    _, bins, patches = ax.hist(sizes, 32)
    ax.set_xlabel("Size in pixel")
    ax.set_ylabel("Count")
    plt.show()

    sizes = sizes[sizes <= bins[1]]
    fig, ax = plt.subplots()
    _, bins, patches = ax.hist(sizes, 32)
    print(bins)
    ax.set_xlabel("Size in pixel")
    ax.set_ylabel("Count")
    plt.show()


def filter_by_size(table, size_threshold):
    table = table.loc[table['n_pixels'] > size_threshold]
    return table


def compute_offsets(table):
    df = table[['anchor_x', 'anchor_y', 'anchor_y']]
    pos = df.values
    offsets = np.linalg.norm(pos, axis=1)
    return offsets


def plot_offsets(table):
    offsets = compute_offsets(table)
    fig, ax = plt.subplots()
    ax.hist(offsets, 32)
    ax.set_xlabel("Offset in microns")
    ax.set_ylabel("Count")
    plt.show()


def filter_by_offset(table, offset_threshold):
    offsets = compute_offsets(table)
    table = table.loc[offsets > offset_threshold]
    return table
