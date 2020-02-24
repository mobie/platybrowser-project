from elf.io import open_file
from pybdv.converter import make_bdv


def copy_midline():
    p = '../../EM-Prospr/kim-midline-2-new.h5'
    resolution = [.5, .5, .5]
    with open_file(p, 'r') as f:
        ds = f['t00000/s00/0/cells']
        mid = ds[:]

    out = '../data/rawdata/sbem-6dpf-1-whole-segmented-midline.n5'
    ds_factors = 3 * [[2, 2, 2]]
    make_bdv(mid, out, resolution=resolution, unit='micrometer',
             downscale_factors=ds_factors, convert_dtype=False,
             chunks=(96,) * 3, n_threads=4)


if __name__ == '__main__':
    copy_midline()
