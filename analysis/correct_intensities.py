#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import os
import numpy as np
import h5py
import z5py
import vigra
from scripts.transformation import intensity_correction


def combine_mask():
    tmp_folder = './tmp_intensity_correction'
    os.makedirs(tmp_folder, exist_ok=True)

    mask_path1 = '../data/rawdata/sbem-6dpf-1-whole-mask-inside.h5'
    mask_path2 = '../data/rawdata/sbem-6dpf-1-whole-mask-resin.h5'

    print("Load inside mask ...")
    with h5py.File(mask_path1, 'r') as f:
        key = 't00000/s00/0/cells'
        mask1 = f[key][:].astype('bool')
    print("Load resin mask ..")
    with h5py.File(mask_path2, 'r') as f:
        key = 't00000/s00/1/cells'
        mask2 = f[key][:]

    print("Resize resin mask ...")
    mask2 = vigra.sampling.resize(mask2.astype('float32'), mask1.shape, order=0).astype('bool')
    mask = np.logical_or(mask1, mask2).astype('uint8')

    out_path = 'mask.n5'
    out_key = 'data'
    with z5py.File(out_path) as f:
        f.create_dataset(out_key, data=mask, compression='gzip', n_threads=8)


def correct_intensities(target='slurm', max_jobs=250):
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


if __name__ == '__main__':
    # combine_mask()
    correct_intensities('local', 32)
