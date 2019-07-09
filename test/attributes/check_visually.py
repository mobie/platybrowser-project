import h5py
import vigra
import numpy as np
import pandas as pd
from cremi_tools.viewer.volumina import view


def check_region_mapping():
    t = '../../data/0.1.1/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv'
    table = pd.read_csv(t, sep='\t')
    label_ids = table['label_id'].values
    head_ids = table['head'].values
    head_ids = label_ids[head_ids == 1]

    p1 = '../../data/0.1.1/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    p2 = '../../data/rawdata/prospr-6dpf-1-whole-segmented-Head.h5'

    with h5py.File(p1, 'r') as f:
        ds = f['t00000/s00/4/cells']
        seg = ds[:]

    with h5py.File(p2, 'r') as f:
        ds = f['t00000/s00/0/cells']
        head = ds[:]

    shape = head.shape
    seg = vigra.sampling.resize(seg.astype('float32'), shape, order=0).astype('uint32')
    head_mask = np.isin(seg, head_ids).astype('uint32')

    view([head, head_mask, seg])


if __name__ == '__main__':
    check_region_mapping()
