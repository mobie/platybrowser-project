import os
import z5py
import h5py
import numpy as np
import pandas as pd
import nifty.tools as nt
from pybdv import make_bdv
from mmpb.attributes.base_attributes import base_attributes


# Segmentation version: 0.2.1
def write_assignments():
    tab = pd.read_csv('./head_ganglions_table_v1.csv', sep='\t')
    g = tab['Head_ganglion_1'].values.astype('uint32')
    print(g.shape)

    with z5py.File('ganglion.n5') as f:
        f.create_dataset('assignments', data=g, compression='gzip', chunks=(10000,))


def write_ganglia():
    tab = pd.read_csv('./head_ganglions_table_v1.csv', sep='\t')
    g = tab['Head_ganglion_1'].values.astype('uint32')
    g[0] = 0
    max_id = int(g.max())
    print("Max-id:", max_id)

    seg_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.2.1',
                            'segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    seg_key = 't00000/s00/3/cells'

    out_path = './sbem-6dpf-1-whole-segmented-ganglia-labels.h5'

    print("Reading segmentation ...")
    with h5py.File(seg_path, 'r') as f:
        ds = f[seg_key]
        seg = ds[:].astype('uint32')

    print("To ganglion segmentation ...")
    seg = nt.take(g, seg)
    seg = seg.astype('int16')

    print("Writing segmentation ...")
    n_scales = 4
    res = [.2, .16, .16]
    downscale_factors = n_scales * [[2, 2, 2]]
    make_bdv(seg, out_path, downscale_factors,
             resolution=res, unit='micrometer')
    with h5py.File(out_path) as f:
        ds = f['t00000/s00/0/cells']
        ds.attrs['maxId'] = max_id


def make_table():
    input_path = './sbem-6dpf-1-whole-segmented-ganglia-labels.h5'
    input_key = 't00000/s00/0/cells'
    output_path = './default.csv'
    resolution = [.2, .16, .16]
    tmp_folder = './tmp_ganglia'
    target = 'local'
    max_jobs = 48
    base_attributes(input_path, input_key, output_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=False)


def make_overlap_table():
    from mmpb.attributes.util import write_csv, node_labels
    tmp_folder = './tmp_ganglia'
    target = 'slurm'
    max_jobs = 200

    seg_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.5',
                            'segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    seg_key = 't00000/s00/0/cells'

    input_path = './sbem-6dpf-1-whole-segmented-ganglia-labels.h5'
    input_key = 't00000/s00/0/cells'

    prefix = 'ganglia'

    ganglia_labels = node_labels(seg_path, seg_key,
                                 input_path, input_key, prefix,
                                 tmp_folder, target, max_jobs)

    n_labels = len(ganglia_labels)
    n_ganglia = ganglia_labels.max() + 1
    assert n_ganglia == 19
    col_names = ['label_id', 'ganglion_id']
    n_cols = len(col_names)

    reg_table = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.5',
                             'tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv')
    reg_table = pd.read_csv(reg_table, sep='\t')
    print(reg_table.columns)
    assert len(reg_table) == len(ganglia_labels)

    have_ganglia = ganglia_labels > 0

    head_ids = reg_table['head'].values > 0
    head_ids = np.logical_and(head_ids, ~have_ganglia)
    print("Rest of head:", head_ids.sum())
    ganglia_labels[head_ids] = n_ganglia

    have_labels = ganglia_labels > 0
    muscle_ids = reg_table['muscle'].values > 0
    muscle_ids = np.logical_and(muscle_ids, have_labels)
    print("Muscles in labels:", muscle_ids.sum())
    ganglia_labels[muscle_ids] = 0

    table = np.zeros((n_labels, n_cols))
    table[:, 0] = np.arange(n_labels)
    table[:, 1] = ganglia_labels

    out_path = './ganglia_ids.csv'
    write_csv(out_path, table, col_names)


if __name__ == '__main__':
    # write_ganglia()
    # make_table()
    make_overlap_table()
