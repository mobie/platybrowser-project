import h5py
import numpy as np
from pybdv.util import get_key
from mmpb.util import is_h5_file


def number_of_voxels():
    p = '../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    is_h5 = is_h5_file(p)
    key = get_key(is_h5, setup_id=0, time_point=0, scale=0)
    with h5py.File(p, 'r') as f:
        ds = f[key]
        shape = ds.shape
    n_vox = np.prod(list(shape))
    print("Number of voxel:")
    print(n_vox)
    print("corresponds to")
    print(float(n_vox) / 1e12, "TVoxel")


def animal():
    p = '../data/rawdata/sbem-6dpf-1-whole-mask-inside.n5'
    is_h5 = is_h5_file(p)
    key = get_key(is_h5, setup_id=0, time_point=0, scale=0)
    with h5py.File(p, 'r') as f:
        mask = f[key][:]

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
