import numpy as np
from concurrent import futures

import z5py
import vigra

from skimage.exposure import histogram
from pybdv.converter import make_bdv
from elf.wrapper.resized_volume import ResizedVolume
from heimdall import view


def threshold_otsu(image, nbins=256):
    hist, bin_centers = histogram(image.ravel(), nbins, source_range='image')
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def make_resin_mask_2d(z=None, scale=3):
    path = '../data.n5'
    raw_key = 'volumes/raw-samplexy/s%i' % scale
    mask_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-inside.n5'
    mask_key = 'setup0/timepoint0/s0'

    n_threads = 8

    f = z5py.File(path)
    ds_raw = f[raw_key]

    f_mask = z5py.File(mask_path)
    ds = f_mask[mask_key]
    ds.n_threads = n_threads
    mask = ds[:]
    mask = ResizedVolume(mask, ds_raw.shape, order=0)

    size_thresh = 5000

    def mask_2d(z):
        print(z, "/", ds_raw.shape[0])
        raw = ds_raw[z]
        maskz = np.logical_not(mask[z])
        maskz = np.logical_or(maskz, raw == 0)
        maskz = np.logical_or(maskz, raw == 255)

        # run otsu on the remaining data to get rid of the embedded silver
        masked = raw[maskz]
        thresh = threshold_otsu(masked)
        maskz = np.logical_and(maskz, raw > thresh)

        # get rid of upper quantile
        masked = raw[maskz]
        # thresh = threshold_otsu(masked)
        thresh = np.quantile(masked, .9)
        maskz = np.logical_and(maskz, raw < thresh)

        # only keep the biggest component
        ccs = vigra.analysis.labelImageWithBackground(maskz.astype('uint32'))
        ids, sizes = np.unique(ccs, return_counts=True)
        ids, sizes = ids[1:], sizes[1:]
        keep_ids = ids[sizes > size_thresh]
        maskz = np.isin(ccs, keep_ids)

        maskz = maskz.astype('uint8') * 255
        return maskz

    if z is not None:
        resin_mask = mask_2d(z)
        raw = ds_raw[z]
        mask = mask[z]
        view(raw, mask, resin_mask)

    else:
        print("Compute mask")
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(mask_2d, z) for z in range(ds_raw.shape[0])]
            res = [t.result() for t in tasks]

        resin_mask = np.concatenate([re[None] for re in res], axis=0)
        print(resin_mask.shape)

        # save as bdv, why not
        # s0: .025, .02, .02
        # s1: .025, .08, .08
        # s2: .025, .16, .16
        # s3: .025, .32, .32
        res = [.025, .32, .32]
        make_bdv(resin_mask, 'sbem-6dpf-1-whole-segmented-resin.n5',
                 downscale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 resolution=res, unit='micrometer', downscale_mode='min')


def export_raw(scale=3):
    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    key_raw = 'volumes/raw_sample_xy/s%i' % scale

    f = z5py.File(path)
    ds = f[key_raw]
    ds.n_threads = 32
    raw = ds[:]
    res = [.025, .32, .32]
    make_bdv(raw, 'em-raw-samplexy.h5',
             downscale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
             resolution=res, unit='micrometer', downscale_mode='mean')


if __name__ == '__main__':
    # working slice: 1676
    # not working: 1620
    # cc issues:7158
    # make_resin_mask_2d(7158)
    make_resin_mask_2d()
    # export_raw()
