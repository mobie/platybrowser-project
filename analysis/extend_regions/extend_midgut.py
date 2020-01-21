import h5py
from pybdv import make_bdv
import pandas as pd
import numpy as np


def make_midgut_volume():
    table_path = './corrected_regions_1.csv'
    table = pd.read_csv(table_path, sep='\t')
    label_ids = table['label_id'].values
    midgut = table['midgut'].values
    assert len(label_ids) == len(midgut)
    assert np.array_equal(np.unique(midgut), np.array([0, 1]))
    midgut_ids = label_ids[midgut == 1]

    scale = 4
    seg_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    key = 't00000/s00/%i/cells' % scale
    res = [0.4, 0.32, 0.32]

    with h5py.File(seg_path, 'r') as f:
        seg = f[key][:]

    midgut_seg = 255 * np.isin(seg, midgut_ids).astype('int8')
    out_path = './sbem-6dpf-1-whole-segmented-midgut.h5'
    factors = 3 * [[2, 2, 2]]
    make_bdv(midgut_seg, out_path, factors, resolution=res, unit='micrometer',
             convert_dtype=False)


if __name__ == '__main__':
    make_midgut_volume()
