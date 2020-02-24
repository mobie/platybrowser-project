import h5py
import numpy as np
import pandas as pd
from skimage.io import imread


def get_volume(path, level):
    """ extract dataset from big data viewer file at specified level"""
    with h5py.File(path, 'r') as f:
        # get specified layer in bdv file
        data = f['t00000/s00/' + str(level) + '/cells']
        img_array = data[:]

    return img_array


def z_profile(arr, method='median', mask=False):
    """
    Calculate mean / median for every z slice in the array and return as a list
    Ignores values of 0 and 255

    Args:
        arr [np.array] - image to process
        method [str] - 'mean' or 'median'
        mask [np.array] - binary mask (0/1) same dimensions as arr - only this region will be considered

    Returns:
        [list] - one entry for each slice
    """

    result = []
    for z, sl in enumerate(arr):
        if mask is False:
            vals = sl[sl != 0]
            vals = vals[vals != 255]

        else:
            mask_slice = mask[z, :, :]
            vals = sl[mask_slice != 0]

        #  If empty, just return zero
        if vals.size == 0:
            print('empty slice' + str(z))
            result.append(0)
            continue

        if method == 'median':
            med_intensity = np.median(vals)
        elif method == 'mean':
            med_intensity = np.mean(vals, dtype=np.float64)
        result.append(med_intensity)
    return result


def profiles_for_raw(raw, paths_to_masks, names_of_masks, zmin=False, zmax=False, exclude_zeros=True):
    """
    Calculates the mean / median value of the raw for each slice (within each of the provided masks)

    Args:
        raw [np.array] - image to process
        paths_to_masks [listlike of str] - paths to locations of tiff files of masks (same size as raw)
        names_of_masks [listlike of str] - names of masks provided
        zmin [bool or int] - if False, use full range, otherwise use provided int as the minimum z value
        zmax [bool or int] - if False, use full range, otherwise use provided int as the maximum z value
        exlude_zeros [bool] - whether to exclude zero values from calculations

    Returns:
        [pd.DataFrame] - nrow == number of z slices, and 2 columns for each mask (median & mean)
    """
    names = names_of_masks
    paths = paths_to_masks

    median_profiles = []
    mean_profiles = []
    for path, name in zip(paths, names):
        print(name)
        exclude = np.copy(raw)
        arr = imread(path)

        if zmin is not False:
            arr = arr[zmin:zmax, :, :]

        exclude[arr == 0] = 0
        if exclude_zeros:
            med_profile = z_profile(exclude, 'median')
            median_profiles.append(med_profile)

            mean_profile = z_profile(exclude, 'mean')
            mean_profiles.append(mean_profile)

        else:
            med_profile = z_profile(exclude, 'median', arr)
            median_profiles.append(med_profile)

            mean_profile = z_profile(exclude, 'mean', arr)
            mean_profiles.append(mean_profile)

    medians = pd.DataFrame(median_profiles).transpose()
    means = pd.DataFrame(mean_profiles).transpose()

    res = pd.concat([medians, means], axis=1)
    names_med = [name + '_median' for name in names]
    names_mea = [name + '_mean' for name in names]
    res.columns = names_med + names_mea

    return res


if __name__ == '__main__':
    pass
