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


def check_nucleus_mapping(scale):
    # t = '../../data/0.1.1/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv'
    t = './tmp/table-test.csv'
    table = pd.read_csv(t, sep='\t')
    label_ids = table['label_id'].values
    nucleus_ids = table['nucleus_id'].values
    label_ids = label_ids[nucleus_ids != 0]
    nucleus_ids = np.unique(nucleus_ids)[1:]

    print("Found %i cells with nuclei" % len(label_ids))

    pr = '../../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    ps = '../../data/0.1.1/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    pn = '../../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'

    print("Load raw")
    with h5py.File(pr, 'r') as f:
        ds = f['t00000/s00/%i/cells' % scale]
        raw = ds[:]

    print("Load seg")
    with h5py.File(ps, 'r') as f:
        ds = f['t00000/s00/%i/cells' % (scale - 1)]
        seg = ds[:]
    assert seg.shape == raw.shape

    print("Load nuc")
    with h5py.File(pn, 'r') as f:
        ds = f['t00000/s00/%i/cells' % (scale - 3)]
        nuc = ds[:]
    assert nuc.shape == seg.shape

    print("Mask cells")
    mapped_mask = np.isin(seg, label_ids)
    seg_masked = seg.copy()
    seg_masked[np.logical_not(mapped_mask)] = 0

    print("Mask nuclei")
    mapped_mask = np.isin(nuc, nucleus_ids)
    nuc_masked = nuc.copy()
    nuc_masked[np.logical_not(mapped_mask)] = 0

    view([raw, seg, seg_masked, nuc, nuc_masked],
         ['raw', 'seg', 'seg-masked', 'nuc', 'nuc-masked'])


if __name__ == '__main__':
    # check_region_mapping()
    check_nucleus_mapping(5)
