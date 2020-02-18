import numpy as np
import pandas as pd
from elf.io import open_file
from pybdv import make_bdv
from pybdv.util import get_key


def make_midgut_volume():
    table_path = './corrected_regions_1.csv'
    table = pd.read_csv(table_path, sep='\t')
    label_ids = table['label_id'].values
    midgut = table['midgut'].values
    assert len(label_ids) == len(midgut)
    assert np.array_equal(np.unique(midgut), np.array([0, 1]))
    midgut_ids = label_ids[midgut == 1]

    scale = 4
    seg_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-cells.n5'
    res = [0.4, 0.32, 0.32]

    key = get_key(False, time_point=0, setup_id=0, scale=scale)
    with open_file(seg_path, 'r') as f:
        ds = f[key]
        seg = ds[:]
        chunks = ds.chunks

    midgut_seg = 255 * np.isin(seg, midgut_ids).astype('int8')
    out_path = './sbem-6dpf-1-whole-segmented-midgut.n5'
    factors = 3 * [[2, 2, 2]]
    make_bdv(midgut_seg, out_path, factors, resolution=res, unit='micrometer',
             convert_dtype=False, chunks=chunks)


if __name__ == '__main__':
    make_midgut_volume()
