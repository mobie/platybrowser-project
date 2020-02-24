#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import numpy as np
import h5py
import vigra

from scipy.ndimage.morphology import binary_dilation
from mmpb.transformation import intensity_correction
from pybdv import make_bdv

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


def correct_intensities(target='slurm', max_jobs=250):
    raw_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-raw.h5')
    tmp_folder = './tmp_intensity_correction'

    # TODO need to check which mask this is and then take it from the data folder
    mask_path = 'mask.h5'
    mask_key = 't00000/s00/0/cells'

    out_path = 'em-raw-wholecorrected.n5'
    trafo = './intensity_correction_parameters.json'

    resolution = [0.025, 0.01, 0.01]
    intensity_correction(raw_path, out_path, mask_path, mask_key,
                         trafo, tmp_folder, resolution,
                         target=target, max_jobs=max_jobs)


def combine_mask():
    tmp_folder = './tmp_intensity_correction'
    os.makedirs(tmp_folder, exist_ok=True)

    mask_path1 = '../data/rawdata/sbem-6dpf-1-whole-mask-inside.h5'
    mask_path2 = '../data/rawdata/sbem-6dpf-1-whole-mask-resin.h5'

    print("Load inside mask ...")
    with h5py.File(mask_path1, 'r') as f:
        key = 't00000/s00/0/cells'
        mask1 = f[key][:].astype('bool')
        mask1 = binary_dilation(mask1, iterations=4)
    print("Load resin mask ..")
    with h5py.File(mask_path2, 'r') as f:
        key = 't00000/s00/1/cells'
        mask2 = f[key][:]

    print("Resize resin mask ...")
    mask2 = vigra.sampling.resize(mask2.astype('float32'), mask1.shape, order=0).astype('bool')
    mask = np.logical_or(mask1, mask2).astype('uint8')

    res = [.4, .32, .32]
    ds_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    make_bdv(mask, 'mask.h5', ds_factors,
             resolution=res, unit='micrometer')


def make_extrapolation_mask():
    z0 = 800  # extrapolation for z < z0
    z1 = 9800  # extraplation for z > z1

    ref_path = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    ref_scale = 4
    ref_key = 't00000/s00/%i/cells' % ref_scale

    with h5py.File(ref_path, 'r') as f:
        shape = f[ref_key].shape
    mask = np.zeros(shape, dtype='uint8')

    # adapt to the resolution level
    z0 //= (2 ** (ref_scale - 1))
    z1 //= (2 ** (ref_scale - 1))
    print(z0, z1)

    mask[:z0] = 255
    mask[z1:] = 255
    print(mask.min(), mask.max())

    scales = 3 * [[2, 2, 2]]
    res = [.2, .16, .16]

    out_path = './extrapolation_mask'
    make_bdv(mask, out_path, downscale_factors=scales,
             downscale_mode='nearest',
             resolution=res, unit='micrometer',
             convert_dtype=False)


if __name__ == '__main__':
    correct_intensities()
    # make_extrapolation_mask()
