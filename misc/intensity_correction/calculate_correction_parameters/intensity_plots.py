import numpy as np
import matplotlib.pyplot as plt
import os


def profile_plots(profiles, folder):
    """
    Plot graphs of the z profile of various stats in the data

    Args:
        profiles [pd.DataFrame] -  nrow == number of z slices, and a column for each statistic to plot
        folder [str] - path to folder to save result
    """

    # plots for all corrected as lines
    plt.figure(figsize=(30, 20))
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('Z slice')
    plt.ylabel('Median intensity')
    plt.title('z profile')
    plt.xticks(rotation=90)
    for col_name in profiles.columns:
        result = profiles[col_name]
        result = np.array(result)
        result_no_zeros = result[result != 0]
        zs = np.arange(0, len(result))[result != 0]
        plt.plot(zs, result_no_zeros, label=col_name)
    plt.legend()
    plt.savefig(os.path.join(folder, 'plots', 'all_together.png'))
    plt.close()


def profile_plots_vs_uncorrected(profiles, uncorrected_profiles, folder, lower=5000, upper=6000):
    """
    Plots for various stats vs z_slice (corrected and uncorrected to compare), and then for a zoomed in z range

    Args:
        profiles [pd.DataFrame] - profiles from correction run to plot (resin in first column) - nrow == number z
            slices, one column per statistic
        uncorrected_profiles [pd.DataFrame] - profiles from before correction to plot (resin in first column) -
            nrow == number of z slices, one column per statistic
        folder [str] - path to folder to save result to
        lower [int] - lower z value to use for zoomed graphs
        upper [int] - upper z value to use for zoomed graphs
    """

    # plot for corrected and uncorrected
    for i in range(1, profiles.shape[1]):
        to_plot = [profiles.iloc[:, 0], profiles.iloc[:, i], uncorrected_profiles.iloc[:, 0],
                   uncorrected_profiles.iloc[:, i]]
        profile_names = [profiles.columns[0], profiles.columns[i], 'uncorrected_' + uncorrected_profiles.columns[0],
                         'uncorrected_' + uncorrected_profiles.columns[i]]

        plt.figure(figsize=(30, 20))
        plt.rcParams.update({'font.size': 14})
        plt.style.use('tableau-colorblind10')
        plt.xlabel('Z slice')
        plt.ylabel('Median intensity')
        plt.title('z profile')
        plt.xticks(rotation=90)
        for j, prof in enumerate(to_plot):
            result = np.array(prof)
            result_no_zeros = result[result != 0]
            zs = np.arange(0, len(result))[result != 0]
            plt.plot(zs, result_no_zeros, label=profile_names[j])
        plt.legend()
        plt.savefig(os.path.join(folder, 'plots', 'all_together_with_uncorrected' + str(profiles.columns[i]) + '.png'),
                    dpi=300)
        plt.close()

    # plot for corrected and uncorrected with ticks
    for i in range(1, profiles.shape[1]):
        to_plot = [profiles.iloc[:, 0], profiles.iloc[:, i], uncorrected_profiles.iloc[:, 0],
                   uncorrected_profiles.iloc[:, i]]
        profile_names = [profiles.columns[0], profiles.columns[i], 'uncorrected_' + uncorrected_profiles.columns[0],
                         'uncorrected_' + uncorrected_profiles.columns[i]]
        symbols = ['b-', 'r-', 'b:', 'r:']

        # plots for smaller z range
        zs = np.arange(lower, upper)
        z_cuts = np.array_split(zs, 4)

        for cut in z_cuts:

            plt.figure(figsize=(30, 20))
            plt.rcParams.update({'font.size': 12})
            plt.xlabel('Z slice')
            plt.ylabel('Median intensity')
            plt.title('z profile')
            plt.xticks(np.arange(0, profiles.shape[0] + 1, 1.0))
            plt.xticks(rotation=90)
            for j, prof in enumerate(to_plot):
                result = np.array(prof)
                result_no_zeros = result[result != 0]
                zs = np.arange(0, len(result))[result != 0]
                plt.plot(zs, result_no_zeros, symbols[j], label=profile_names[j])
            plt.xlim(np.min(cut), np.max(cut))
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(folder, 'plots', 'all_together_with_uncorrected' + str(profiles.columns[i]) + str(
                np.min(cut)) + '.png'), dpi=300)
            plt.close()


def plot_offset(parameters, folder):
    """
    Plots the mult and offset from that correction as a graph over z

    Args:
        parameters [pd.DataFrame] - calculated parameters from correction run - containing columns for mult and offset
        folder [str] - path to folder to save results to
    """

    offset = parameters['offset']
    mult = parameters['mult']
    names = ['offset', 'mult']

    for val, name in zip([offset, mult], names):
        plt.figure(figsize=(30, 20))
        plt.rcParams.update({'font.size': 14})
        plt.style.use('tableau-colorblind10')
        plt.xlabel('Z slice')
        plt.ylabel(name)
        result = np.array(val)
        result_no_zeros = result[result != 0]
        zs = np.arange(0, len(result))[result != 0]
        plt.plot(zs, result_no_zeros)
        plt.savefig(os.path.join(folder, 'plots', str(name) + '.png'))
        plt.close()


if __name__ == '__main__':
    pass
