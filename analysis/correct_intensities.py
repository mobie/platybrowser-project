#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import os
import numpy as np
import h5py
import z5py
import vigra

from scipy.ndimage.morphology import binary_dilation
from scripts.transformation import intensity_correction
from pybdv import make_bdv


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


def correct_intensities_test(target='local', max_jobs=32):
    raw_path = '../../EM-Prospr/em-raw-samplexy.h5'
    tmp_folder = './tmp_intensity_correction'

    mask_path = 'mask.n5'
    mask_key = 'data'

    out_path = 'em-raw-samplexy-corrected.h5'

    # trafo = './new_vals.csv'
    trafo = './new_vals.json'

    resolution = [0.025, 0.32, 0.32]
    intensity_correction(raw_path, out_path, mask_path, mask_key,
                         trafo, tmp_folder, resolution,
                         target=target, max_jobs=max_jobs)


def correct_intensities(target='slurm', max_jobs=250):
    raw_path = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    tmp_folder = './tmp_intensity_correction'

    mask_path = 'mask.h5'
    mask_key = 't00000/s00/0/cells'

    out_path = 'em-raw-wholecorrected.h5'

    # trafo = './new_vals.csv'
    trafo = './new_vals.json'
    tmp_path = '/g/kreshuk/pape/Work/platy_tmp.n5'

    resolution = [0.025, 0.01, 0.01]
    intensity_correction(raw_path, out_path, mask_path, mask_key,
                         trafo, tmp_folder, resolution,
                         target=target, max_jobs=max_jobs,
                         tmp_path=tmp_path)


if __name__ == '__main__':
    # combine_mask()
    correct_intensities('local', 64)
