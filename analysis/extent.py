import h5py
import numpy as np


def number_of_voxels():
    p = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    with h5py.File(p, 'r') as f:
        ds = f['t00000/s00/0/cells']
        shape = ds.shape
    n_vox = np.prod(list(shape))
    print("Number of voxel:")
    print(n_vox)
    print("corresponds to")
    print(float(n_vox) / 1e12, "TVoxel")


def animal():
    p = '../data/rawdata/sbem-6dpf-1-whole-mask-inside.h5'
    with h5py.File(p, 'r') as f:
        mask = f['t00000/s00/0/cells'][:]

    bb = np.where(mask > 0)
    mins = [b.min() for b in bb]
    maxs = [b.max() for b in bb]
    size = [ma - mi for mi, ma in zip(mins, maxs)]

    print("Animal size in pixel:")
    print(size)
    res = [.4, .32, .32]

    size = [si * re for si, re in zip(size, res)]
    print("Animal size in micron:")
    print(size)


if __name__ == '__main__':
    number_of_voxels()
    animal()
