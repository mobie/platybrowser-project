import pandas as pd
import numpy as np
from z_profile_functions_mid import profiles_for_raw, get_volume
from z_profile_functions_range import profiles_for_raw_range
import intensity_plots
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from correction_techniques import normalise_spread_mid, normalise_mult_add, fix_ends
from convert_to_bdv import convert_to_bdv


def profile_plots(profiles, folder, method, lower=6000, upper=7000):
    """
    Plot graphs of the z profile of various stats in the data

    Args:
        profiles [pd.DataFrame] - nrow == number of z slices, and a column for each statistic to plot
        folder [str] - path to folder to save plots to
        method [str] - what to put on teh y axis e.g. 'median', 'mean'..
        lower [int] - lower z value to use for zoomed in plots
        upper [int] - upper z value to use for zoomed in plots
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # plots for all corrected together as lines for full z range
    plt.figure(figsize=(30, 20))
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('Z slice')
    plt.ylabel(method)
    plt.title('z profile')
    for col_name in profiles.columns:
        result = profiles[col_name]
        result = np.array(result)
        result_no_zeros = result[result != 0]
        zs = np.arange(0, len(result))[result != 0]
        plt.plot(zs, result_no_zeros, label=col_name)
    plt.legend()
    plt.savefig(os.path.join(folder, 'all_together.png'))
    plt.close()

    # zoomed in plots for a certain z range (a range of about 1000 total works well here)
    zs = np.arange(lower, upper)
    z_cuts = np.array_split(zs, 4)

    for cut in z_cuts:

        plt.figure(figsize=(30, 20))
        plt.style.use('tableau-colorblind10')
        plt.rcParams.update({'font.size': 12})
        plt.xlabel('Z slice')
        plt.ylabel(method)
        plt.title('z profile')
        plt.xticks(np.arange(0, profiles.shape[0] + 1, 1.0))
        plt.xticks(rotation=90)
        for col_name in profiles.columns:
            result = profiles[col_name]
            result = np.array(result)
            result_no_zeros = result[result != 0]
            zs = np.arange(0, len(result))[result != 0]
            plt.plot(zs, result_no_zeros, label=col_name)
        plt.xlim(np.min(cut), np.max(cut))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(folder, 'all_together_z_range_' + str(
            np.min(cut)) + '.png'), dpi=300)
        plt.close()


def process_correction_result(correction_folder, name, paths_to_masks, names_of_masks, path_to_uncorrected_profiles,
                              lows=[6000, 7000], highs=[7000, 8000]):
    """
    Make various intensity plots to assess correction result

    Args:
        correction_folder [str] - path to folder where correction results were saved
        name [str] - name of correction method
        paths_to_masks [listlike of str] - paths to masks you would like to plot median profiles for
        names_of_masks [listlike of str] - names of provided masks
        path_to_uncorrected_profiles [str] - path to table (.csv) of z profiles before correction
        lows [listlike of ints] - minimum z values to use for zoomed graphs
        highs [listlike of ints] - maximum z values to use for zoomed graphs
    """

    # get new median / mean profiles for provided masks
    raw = imread(os.path.join(correction_folder, name, 'corrected_raw.tiff'))
    res = profiles_for_raw(raw, paths_to_masks, names_of_masks)
    res.to_csv(os.path.join(correction_folder, name, 'corrected_profiles.csv'), index=False, sep='\t')

    # get uncorrected z profiles (median)
    uncorrected = pd.read_csv(path_to_uncorrected_profiles, sep='\t')
    uncorrected = uncorrected.filter(regex=("median"))

    os.mkdir(os.path.join(correction_folder, name, 'plots'))

    # get corrected z profiles (median)
    profile = pd.read_csv(os.path.join(correction_folder, name, 'corrected_profiles.csv'), sep='\t')
    cut_table = profile.filter(regex=("median"))

    # make summary plots of stats after correction
    intensity_plots.profile_plots(cut_table, os.path.join(correction_folder, name))

    # make summary plots comparing before and after correction, for various z ranges
    for l, h in zip(lows, highs):
        intensity_plots.profile_plots_vs_uncorrected(cut_table, uncorrected, os.path.join(correction_folder, name),
                                                     lower=l, upper=h)

    # If present, plot the mult & offset values used for correction [linear correction of mult*x + offset]
    res_path = os.path.join(correction_folder, name, 'result.csv')
    if os.path.exists(res_path):
        parameters = pd.read_csv(res_path, sep='\t')
        intensity_plots.plot_offset(parameters, os.path.join(correction_folder, name))

    # make bdv version of final result for easy browsing
    convert_to_bdv(correction_folder, name)

    return


if __name__ == '__main__':
    # raw data - downsampled in xy, but not in z to a resolution of 0.32x0.32x0.025 microns
    raw = get_volume('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Raw\\em-raw-samplexy.h5', 0)

    # segmentations at same scale as the downsampled raw
    resin = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Segmentations\\resin_seg.tiff'
    muscle = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Segmentations\\muscle_seg.tiff'
    neuropil = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Segmentations\\neuropil_seg.tiff'
    nuc = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Segmentations\\nuclei_seg.tiff'
    platy_in = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Segmentations\\shell_seg.tiff'

    # Calculate mean / median of every z slice (for area covered by given segmentations)
    res = profiles_for_raw(raw, [resin, muscle, nuc, neuropil, platy_in],
                           ['resin', 'muscle', 'nuc', 'neuropil', 'platy_in'])
    res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\uncorrected_profiles.csv',
        index=False, sep='\t')

    # Calculate standard deviation & 5% quantiles for every z slice (for area covered by given segmentation)
    res = profiles_for_raw_range(raw, [resin, platy_in, muscle, nuc, neuropil],
                                 ['resin', 'platy_in', 'muscle', 'nuc', 'neuropil'])
    res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\uncorrected_profiles_range.csv',
        index=False, sep='\t')

    # plots of uncorrected mid values & ranges - particularly the median and 5 quantile -
    # to assess current state of data
    uncorrected_mids = pd.read_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\uncorrected_profiles.csv',
        sep='\t')
    uncorrected_range = pd.read_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\uncorrected_profiles_range.csv',
        sep='\t')

    cut_table = uncorrected_mids.filter(regex="median")
    profile_plots(cut_table,
                  'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\median_plots\\',
                  'median')

    cut_table = uncorrected_range.filter(regex="5_quantile")
    profile_plots(cut_table,
                  'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\quantile_5_plots\\',
                  '5_quantile')

    # Perform corrections
    raw_path = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\Raw\\em-raw-samplexy.h5'
    save_folder = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\corrections\\'
    path_to_uncorrected = 'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\uncorrected\\uncorrected_profiles.csv'

    # normalise using the upper 5% quantile as the reference, and the 5% quantile range as the range
    normalise_spread_mid(raw_path, uncorrected_range['platy_in_5_quantile_upper'],
                         uncorrected_range['platy_in_5_quantile'], save_folder, 'platy_in_5_quantile_upper')
    # create various plots to assess how this affected intensity
    process_correction_result(save_folder, 'platy_in_5_quantile_upper', [resin, muscle, nuc, neuropil],
                              ['resin', 'muscle', 'nuc', 'neuropil'], path_to_uncorrected,
                              lows=[0, 1000, 6000, 7000, 10000], highs=[1000, 2000, 7000, 8000, 11000])

    # normalise using the calculated multiplicative values and additions (for direct linear transformation of each slice
    # mult*x + offset) (should be identical, but this is how it will be run on the full data)
    result_5_quantile = pd.read_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\intensity_correction\\final_resin_runs\\corrections\\platy_in_5_quantile_upper\\result.csv',
        sep='\t')
    normalise_mult_add(raw_path, result_5_quantile['mult'], result_5_quantile['offset'],
                       save_folder,
                       'platy_in_5_quantile_upper_mult_offset')
    # create various plots to assess how this affected intensity
    process_correction_result(save_folder, 'platy_in_5_quantile_upper_mult_offset', [resin, muscle, nuc, neuropil],
                              ['resin', 'muscle', 'nuc', 'neuropil'], path_to_uncorrected,
                              lows=[0, 1000, 6000, 7000, 10000], highs=[1000, 2000, 7000, 8000, 11000])

    # Fix the ends of the mult & offset profiles by replacing the values < zmin and values > zmax with the median of the
    # correction factors over the 100 slices next to each cutoff
    # cutoffs were selected by manual inspection of graphs above, for where the resin profile stops being flat and starts
    # increasing
    mult = fix_ends(result_5_quantile['mult'])
    offset = fix_ends(result_5_quantile['offset'])

    # normalise using the fixed result & plot graphs
    normalise_mult_add(raw_path, mult, offset,
                       save_folder,
                       'platy_in_5_quantile_average_ends')
    process_correction_result(save_folder, 'platy_in_5_quantile_average_ends', [resin, muscle, nuc, neuropil],
                              ['resin', 'muscle', 'nuc', 'neuropil'], path_to_uncorrected,
                              lows=[6000, 7000], highs=[7000, 8000])

    # replace the extrapolated value for slice 724 with the calculated one - there is a clear large spike in intensity
    # here, so using the extrapolated value doesn't make sense
    mult = fix_ends(result_5_quantile['mult'])
    offset = fix_ends(result_5_quantile['offset'])
    mult[724] = result_5_quantile['mult'][724]
    offset[724] = result_5_quantile['offset'][724]

    # Save the newer values and run one final time with these values:
    # This is from correction by the upper 5% quantile and the 5% quantile range. Followed by correction of the ends
    # and replacement of one anomalous value for slice 724
    new_vals = pd.DataFrame({'mult': mult, 'offset': offset})
    if not os.path.isdir(os.path.join(save_folder, 'platy_in_5_quantile_average_ends_724')):
        os.mkdir(os.path.join(save_folder, 'platy_in_5_quantile_average_ends_724'))
    new_vals.to_csv(os.path.join(save_folder, 'platy_in_5_quantile_average_ends_724', 'new_vals.csv'), index=False,
                    sep='\t')
    normalise_mult_add(raw_path, mult, offset,
                       save_folder,
                       'platy_in_5_quantile_average_ends_724')
    process_correction_result(save_folder, 'platy_in_5_quantile_average_ends_724', [resin, muscle, nuc, neuropil],
                              ['resin', 'muscle', 'nuc', 'neuropil'], path_to_uncorrected,
                              lows=[6000, 7000], highs=[7000, 8000])
