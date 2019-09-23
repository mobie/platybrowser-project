import numpy as np
import h5py
import vigra
from pybdv import make_bdv


def mask_gut_nuclei(view_mask=False):
    p1 = '../data/0.0.0/images/prospr-6dpf-1-whole-segmented-Stomodeum.h5'
    p2 = '../data/rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.h5'

    with h5py.File(p1, 'r') as f:
        ds = f['t00000/s00/0/cells']
        stomodeum = (ds[:] > 0).astype('float32')

    with h5py.File(p2, 'r') as f:
        ds = f['t00000/s00/2/cells']
        gut = ds[:] == 23

    stomodeum = vigra.sampling.resize(stomodeum, gut.shape, order=0).astype('bool')

    mask = np.logical_and(stomodeum == 0, gut)

    if view_mask:
        from heimdall import view, to_source

        p_nuc = '../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'
        with h5py.File(p_nuc, 'r') as f:
            nuc = f['t00000/s00/2/cells'][:] > 0
        nuc = vigra.sampling.resize(nuc.astype('float32'), gut.shape, order=0).astype('bool')

        view(to_source(gut.astype('uint32'), name='gut-em'),
             to_source(stomodeum.astype('uint32'), name='stomodeum-prospr'),
             to_source(mask.astype('uint32'), name='mask'),
             to_source(nuc.astype('uint32'), name='nuclei'))
    else:
        resolution = [0.4, 0.32, 0.32]
        mask = mask.astype('uint8') * 255
        make_bdv(mask, 'nucleus_exclusion_mask', unit='micrometer',
                 resolution=resolution)


if __name__ == '__main__':
    # shapes()
    mask_gut_nuclei(True)
