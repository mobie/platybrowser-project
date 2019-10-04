#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import os
import json
from concurrent import futures

import numpy as np
import z5py
import h5py
import vigra

from scipy.ndimage.morphology import binary_dilation
from scripts.transformation import intensity_correction
from pybdv import make_bdv, convert_to_bdv


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


# FIXME scale 0 is damaged in the h5 file !
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


def check_chunks():
    import nifty.tools as nt
    path = '/g/kreshuk/pape/Work/platy_tmp.n5'
    key = 'data'
    f = z5py.File(path, 'r')
    ds = f[key]

    shape = ds.shape
    chunks = ds.chunks
    blocking = nt.blocking([0, 0, 0], shape, chunks)

    def check_chunk(block_id):
        print("Check", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlock(block_id)
        chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, chunks))
        try:
            ds.read_chunk(chunk_id)
        except RuntimeError:
            print("Failed:", chunk_id)
            return chunk_id

    print("Start checking", blocking.numberOfBlocks, "blocks")
    with futures.ThreadPoolExecutor(32) as tp:
        tasks = [tp.submit(check_chunk, block_id) for block_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]
    print()
    print(results)
    print()
    with open('./failed_chunks.json', 'w') as f:
        json.dump(results, f)


def make_subsampled_volume():
    p = './em-raw-wholecorrected.h5'
    k = 't00000/s00/2/cells'
    with h5py.File(p, 'r') as f:
        print(f[k].shape)
    # return

    scale_factors = [[2, 2, 2]] * 6
    convert_to_bdv(p, k, 'em-raw-small-corrected.h5', downscale_factors=scale_factors, downscale_mode='mean',
                   resolution=[.05, .04, .04], unit='micrometer')


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
    # correct_intensities('slurm', 125)
    # make_subsampled_volume()
    make_extrapolation_mask()
