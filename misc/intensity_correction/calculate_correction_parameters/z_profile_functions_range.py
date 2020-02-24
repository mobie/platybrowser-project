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


def z_profile(arr, mask=False):
    """
    Calculate standard deviation, median absolute deviation, 5% quantile range
    for every z slice in the array and return as a list
    Ignore zeros and 255s

    Args:
        arr [np.array] - image to process
        mask [np.array] - binary mask (0/1) same dimensions as arr - only this region will be considered

    Returns:
        [tuple of lists] - each list has one entry for every slice (standard deviation, 5 quantile range,
                            lower 5 quantile, upper 5 quantile)
        """
    result_std = []
    result_5_quantile = []
    result_5_quantile_upper = []
    result_5_quantile_lower = []

    for z, sl in enumerate(arr):
        print(z)
        if mask is False:
            vals = sl[sl != 0]
            vals = vals[vals != 255]

        else:
            mask_slice = mask[z, :, :]
            vals = sl[mask_slice != 0]

        #  If empty, just return zero
        if vals.size == 0:
            print('empty slice' + str(z))
            result_std.append(0)
            result_5_quantile.append(0)
            result_5_quantile_upper.append(0)
            result_5_quantile_lower.append(0)
            continue

        result_std.append(np.std(vals))

        ql = np.quantile(vals, 0.05)
        qu = np.quantile(vals, 1 - 0.05)
        result_5_quantile.append(qu - ql)
        result_5_quantile_lower.append(ql)
        result_5_quantile_upper.append(qu)

    return result_std, result_5_quantile, result_5_quantile_lower, result_5_quantile_upper


def profiles_for_raw_range(raw, paths_to_masks, names_of_masks, exclude_zeros=True):
    """
    Calculates various measures of spread of the raw for each z slice (within each of the provided masks)

    Args:
        raw [np.array] - image to process
        paths_to_masks [listlike of str] - paths to locations of tiff files of masks (same size as raw)
        names_of_masks [listlike of str] - names of masks provided
        excldue_zeros [bool] - whether to exclude 0 values from calculation

    Returns:
        [pd.DataFrame] - nrow == number of z slices, and columns for spread of each of the masks
    """

    names = names_of_masks
    paths = paths_to_masks

    all_profiles = []
    for path, name in zip(paths, names):
        print(name)
        exclude = np.copy(raw)
        arr = imread(path)
        exclude[arr == 0] = 0

        if exclude_zeros:
            profiles = z_profile(exclude)
        else:
            profiles = z_profile(exclude, arr)
        all_profiles.extend(profiles)

    res = pd.DataFrame(all_profiles)
    res = res.transpose()
    cols = []

    for name in names:
        for val in ['_std', '_5_quantile', '_5_quantile_lower', '_5_quantile_upper']:
            cols.append(name + val)

    res.columns = cols

    return res


if __name__ == '__main__':
    pass
