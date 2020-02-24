import h5py
import numpy as np
from skimage.io import imsave, imread
import pandas as pd
import os


def get_volume(path, level):
    """ extract dataset from big data viewer file at specified level"""
    with h5py.File(path, 'r') as f:
        data = f['t00000/s00/' + str(level) + '/cells']
        img_array = data[:]

    return img_array


def normalise_spread_mid(path_to_raw, mids, spreads, folder, name):
    """
    Linear transformation of every z slice in the data to match intensity and contrast.
    Will shift to match the provided 'mid' values for each slice to
    the median of all those values. (mids can be any reference of the intensity level in the image e.g. median / mean /
    various quantiles...). Will also adjust the contrast of each z slice so that the 'spreads' values for each slice
    matches to the median of all those values. (spreads can be any reference of the spread of intensity values in the
    image e.g. the standard deviation, the interquartile range...)

    Args:
        path_to_raw [str] - path to the raw data to normalise
        mids [np.array] - mid values to use (len == number of z slices in data) - can be anything you want to use as
            a reference to deal with the shift e.g. median, mean, peak positions etc...
        spreads [np.array] - spread values to use (len == number of z slices in data) - can be anything you want to use
            to deal with the contrast e.g. standard deviation, range, peak widths...
        folder [str] - path to folder to save results
        name [str] - name of correction technique (will be used to name folders and files)

    Returns:
        Saves corrected image and dataframe of values used to specified folder. Dataframe columns contain the mid /
        spread used for each slice, as well as the calculated m & s values (for a linear correction of the form (x-m)/s &
        the multiplicative and offset values (for a linear correction of the form mult*x + offset)
    """
    raw = get_volume(path_to_raw, 0)
    dtype = raw.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    raw = raw.astype('float64')

    #  use median value of the mids / spreads as the reference to which all slices will be matched
    mids_for_ref = mids[mids != 0]
    mid_ref = np.median(mids_for_ref)

    spreads_for_ref = spreads[spreads != 0]
    spread_ref = np.median(spreads_for_ref)

    # record values used
    # m and s are for equation of form (x-m)/s
    # mult and offset are for equation of form mult*x + offset
    # mid is the mid value for each slice used
    # spread is the spread value for each slice used
    m = []
    s = []
    offset = []
    mult = []
    final_mid = []
    final_spread = []

    # Process slices
    for z_index, sl in enumerate(raw):
        print(z_index)

        mid_slice = mids[z_index]
        spread_slice = spreads[z_index]

        # only process if slice isn't empty
        if np.sum(sl) != 0 and mid_slice != 0 and spread_slice != 0:

            final_mid.append(mid_slice)
            final_spread.append(spread_slice)

            m_val = mid_slice - (mid_ref * (spread_slice / spread_ref))
            m.append(m_val)

            s_val = (spread_slice / spread_ref)
            s.append(s_val)

            offset.append(-m_val / s_val)
            mult.append(1 / s_val)

            # Convert the intensities to the target domain
            # but don't mess with the zeros around the platy
            mask_slice = sl.copy() != 0
            sl = np.where(mask_slice, sl - mid_slice, sl)
            sl = np.where(mask_slice, sl / spread_slice, sl)
            sl = np.where(mask_slice, sl * spread_ref, sl)
            sl = np.where(mask_slice, sl + mid_ref, sl)

            raw[z_index] = sl

        else:
            m.append(0)
            s.append(0)
            offset.append(0)
            mult.append(0)
            final_mid.append(0)
            final_spread.append(0)
            raw[z_index] = sl

    # Clip everything that went out of range
    raw[raw < 0] = 0
    raw[raw > 255] = 255

    # dataframe of values
    res = pd.DataFrame({'m': m, 's': s, 'offset': offset, 'mult': mult, 'mid': final_mid, 'spread': final_spread})

    #  Convert back to the original dtype
    raw = raw.astype(dtype)

    if not os.path.isdir(os.path.join(folder, name)):
        os.mkdir(os.path.join(folder, name))
    imsave(os.path.join(folder, name, 'corrected_raw.tiff'), raw)
    res.to_csv(os.path.join(folder, name, 'result.csv'), index=False, sep='\t')

    return raw, res


def normalise_mult_add(path_to_raw, mult, offset, folder, name):
    """
    Normalise each slice by linear transformation - multiplying by mult then adding the offset
    Mult*x + offset

    Args:
        path_to_raw [str] - path to the raw data to normalise
        mult [np.array] - mult values to use (len == number of z slices in raw]
        offset [np.array] - offset values to use (len == number of z slices in raw]
        folder [str] - path to save results to
        name [str] - name of correction technique (will be used to name folders / files)

    Returns:
        Saves corrected image
    """
    raw = get_volume(path_to_raw, 0)
    dtype = raw.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    raw = raw.astype('float64')

    # Process slices
    for z_index, sl in enumerate(raw):
        print(z_index)

        mult_slice = mult[z_index]
        offset_slice = offset[z_index]

        # only process if slice isn't empty
        if np.sum(sl) != 0 and mult_slice != 0:

            # Convert the intensities to the target domain
            # but don't mess with the zeros around the platy
            mask_slice = sl.copy() != 0
            sl = np.where(mask_slice, (sl * mult_slice) + offset_slice, sl)

            raw[z_index] = sl

        else:
            raw[z_index] = sl

    # Clip everything that went out of range
    raw[raw < 0] = 0
    raw[raw > 255] = 255

    #  Convert back to the original dtype
    raw = raw.astype(dtype)

    if not os.path.isdir(os.path.join(folder, name)):
        os.mkdir(os.path.join(folder, name))
    imsave(os.path.join(folder, name, 'corrected_raw.tiff'), raw)

    return raw


def fix_ends(profile, zmin=800, zmax=9800):
    """
    Fix the ends of a profile by replacing the values < zmin and values > zmax with the median of the
    correction factors over the 100 slices next to each cutoff.

    Args:
        profile [np.array] - z profile to correct (len == number of z slices)
        zmin [int] - minimum z cutoff
        zmax [int] - maximum z cutoff
    """

    prof = profile.copy()

    early = prof[zmin:zmin + 100]
    late = prof[zmax - 100:zmax]

    early_to_use = np.median(early)
    late_to_use = np.median(late)

    prof[0:zmin] = early_to_use
    prof[zmax + 1:len(profile)] = late_to_use

    return prof
