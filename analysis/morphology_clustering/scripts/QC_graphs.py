import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from concurrent.futures import ProcessPoolExecutor


def correlation_plot(dataframe, save_path=None):
    """
    All by all (columns) correlation heatmap

    Args:
        dataframe [pd.Dataframe] - dataframe of correlation values
        save_path [str] - path to save result (.pdf), None to just show result
    """
    sns.set(font_scale=0.5)
    sns.set_style('ticks')
    sns.clustermap(dataframe, cmap='mako', figsize=(30, 30))

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)
        plt.close()


# wrapper to call column_against_rest_plot & set up required filepaths to run in parallel
def parallel_all_cols(dataframe, path_root, col):
    path = os.path.join(path_root, 'scatter_plots_%s_vs_all.png' % col)
    column_against_rest_plot(dataframe, col, path)


def column_against_rest_plot(dataframe, column, savepath=None):
    """
    Produces scatter plots for every other column in the dataframe vs the provided column
    Provided column on x, other columns on y

    Args:
        dataframe [pd.Dataframe] - dataframe to plot
        column [str] - name of column
        savepath [str] - where to save the resulting image (.png)
    """
    root_col = column

    # set up number of cols / rows needed for grid of scatter plots
    sqrt_cols = np.sqrt(dataframe.shape[1])
    cols = int(sqrt_cols)
    remaining = dataframe.shape[1] - cols
    rows = 1 + int(np.ceil(remaining / cols))

    plt.figure(figsize=(30, 30))
    for index, col in enumerate(dataframe.columns):
        plt.subplot(rows, cols, index + 1)
        plt.plot(dataframe[root_col], dataframe[col], 'o', ms=1)
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        plt.title(col, fontsize=12, y=1)

    plt.tight_layout()

    if savepath is None:
        plt.show()

    else:
        plt.savefig(savepath)
        plt.close()


def distribution_plot(dataframe, savepath=None):
    """
    Histogram of distribution of each column in the dataframe

    Args:
        dataframe [pd.Dataframe] - dataframe to plot
        savepath [str] - where to save result (.png)
    """

    # set up number of cols / rows needed for grid of histograms
    sqrt_cols = np.sqrt(dataframe.shape[1])
    cols = int(sqrt_cols)
    remaining = dataframe.shape[1] - cols
    rows = 1 + int(np.ceil(remaining / cols))

    plt.figure(figsize=(30, 30))
    for index, col in enumerate(dataframe.columns):
        plt.subplot(rows, cols, index + 1)
        plt.hist(dataframe[col])
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        plt.title(col, fontsize=12, y=1)

    plt.tight_layout()

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.QC):
        os.makedirs(snakemake.output.QC)

    stats = pd.read_csv(snakemake.input.QC_stats, sep='\t')

    # remove label ids
    stats_cut = stats.drop(columns=['label_id_cell', 'label_id_nucleus'])

    # correlation matrix all vs all
    # measure of linear relationship (uses absolute values)
    correlation_p = stats_cut.corr(method='pearson')
    # measure of monotonic relation e.g. x increases when y increases, but not necessarily
    # in a linear way e.g. could be exponential...
    correlation_s = stats_cut.corr(method='spearman')

    path = os.path.join(snakemake.output.QC, 'pearson_correlation_all.pdf')
    correlation_plot(correlation_p, path)
    path = os.path.join(snakemake.output.QC, 'spearman_correlation_all.pdf')
    correlation_plot(correlation_s, path)

    # plot for all vs all scatter plots
    with ProcessPoolExecutor() as e:
        results = list(
            e.map(parallel_all_cols, [stats_cut] * stats_cut.shape[1], [snakemake.output.QC] * stats_cut.shape[1],
                  stats_cut.columns))

    # plot for distributions of each
    path = os.path.join(snakemake.output.QC, 'distribution_all.png')
    distribution_plot(stats_cut, path)


if __name__ == '__main__':
    main()
