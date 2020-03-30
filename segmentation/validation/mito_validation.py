import imageio
import numpy as np
import nifty.tools as nt
import z5py
from tqdm import tqdm


def extract_slice(ds, slice_id, axis, block_shape, prefix):
    if  axis == 0:
        bb = np.s_[slice_id]
    elif axis == 1:
        bb = np.s_[:, slice_id]
    elif axis == 2:
        bb = np.s_[:, :, slice_id]

    data = ds[bb]
    assert data.ndim == 2
    shape = data.shape

    blocking = nt.tools([0, 0], shape, block_shape)
    for block_id in tqdm(range(blocking.numberOfBlocks)):
        block = blocking.getBlock(block_id)
        block_bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        image = data[block_bb]

        if image.sum() == 0:
            continue

        # TODO
        out_path = prefix
        imageio.imwrite(out_path, image)


def make_mito_validation_data(scale=1):
    path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    f = z5py.File(path, 'r')
    key = 'setup0/timepoint0/s%i' % scale
    ds = f[key]
    ds.n_threads = 8

    slices_xy = []
    slices_xz = []

    im_shape = [1024, 1024]
    for slice_id in slices_xy:
        # TODO
        prefix = ''
        extract_slice(ds, slice_id, 0, im_shape, prefix)

    for slice_id in slices_xz:
        # TODO
        prefix = ''
        extract_slice(ds, slice_id, 1, im_shape, prefix)



if __name__ == '__main__':
    make_mito_validation_data()
