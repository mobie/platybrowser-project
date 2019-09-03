import numpy as np
import h5py
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
from heimdall import view
from scripts.attributes.cilia_attributes import (compute_centerline,
                                                 load_seg,
                                                 make_indexable)


def view_centerline(obj, resolution):
    path = compute_centerline(obj, [res * 1000 for res in resolution])
    path = make_indexable(path)
    cline = np.zeros(obj.shape, dtype='bool')
    cline = binary_dilation(cline, iterations=2)
    view(obj.astype('uint32'), cline.astype('uint32'))


def check_lens():
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    resolution = [.025, .01, .01]
    with h5py.File(path) as f:
        ds = f['t00000/s00/0/cells']

        for cid in range(len(table)):
            if cid in (0, 1, 2):
                continue
            obj = load_seg(ds, table, cid, resolution)
            view_centerline(obj, resolution)


if __name__ == '__main__':
    check_lens()
