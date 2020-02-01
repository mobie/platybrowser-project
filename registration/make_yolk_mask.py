import h5py
import numpy as np
import vigra
from pybdv import make_bdv


def make_yolk_mask():
    p = '../data/rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.h5'
    with h5py.File(p, 'r') as f:
        names = f['semantic_names'][:]
        sem_ids = f['semantic_mapping'][:]

        for n, ids in zip(names, sem_ids):
            if n == 'yolk':
                yolk_ids = ids
        ds = f['t00000/s00/2/cells']
        seg = ds[:].astype('uint32')
        mask = np.isin(seg, yolk_ids).astype('uint32')

    p = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    with h5py.File(p, 'r') as f:
        ds = f['t00000/s00/5/cells']
        rshape = ds.shape
    # need to resize due to mis aligned scales
    mask = vigra.sampling.resize(mask.astype('float32'), rshape, order=0).astype('uint8')
    print(mask.shape)
    mask *= 255

    res = [.4, .32, .32]
    n_scales = 3
    scales = n_scales * [[2, 2, 2]]
    make_bdv(mask, './em_yolk_mask', convert_dtype=False,
             unit='micrometer', resolution=res,
             downscale_factors=scales)

    # from heimdall import view
    # view(raw, mask)


if __name__ == '__main__':
    make_yolk_mask()
