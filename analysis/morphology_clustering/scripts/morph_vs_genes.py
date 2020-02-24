import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from concurrent.futures import ProcessPoolExecutor


def correlation_plot(dataframe, save_path=None):
    """
    All by all (columns) correlation heatmap

    Args:
        dataframe [pd.Dataframe] - dataframe of correlations
        save_path [str] - path to save result (.pdf), None to just show result
    """
    # remove any columns that are all nans, these are genes where there
    # is no recorded expression
    dataframe = dataframe.dropna(axis=1, how='all')
    sns.set(font_scale=0.5)
    sns.set_style('ticks')
    sns.clustermap(dataframe, cmap='mako', figsize=(30, 30))

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)
        plt.close()


# wrapper to call column_against_rest_plot & set up required filepaths to run in parallel
def parallel_all_cols(dataframe, path_root, col_name, col_contents):
    path = os.path.join(path_root, 'scatter_plots_%s_vs_all.png' % col_name)
    column_against_rest_plot(dataframe, col_contents, path)


def column_against_rest_plot(dataframe, col_contents, savepath=None):
    """
    Produces scatter plots for every column in dataframe vs the provided col_contents
    Col_contents on x, dataframe columns on y

    Args:
        dataframe [pd.Dataframe] - a dataframe to plot
        col_contents [listlike] - contents of a column
        savepath [str] - where to save the resulting image (.png), None to just show it
    """
    # set up number of cols / rows needed for grid of scatter plots
    sqrt_cols = np.sqrt(dataframe.shape[1])
    cols = int(sqrt_cols)
    remaining = dataframe.shape[1] - cols
    rows = 1 + int(np.ceil(remaining / cols))

    plt.figure(figsize=(30, 30))
    for index, col in enumerate(dataframe.columns):
        plt.subplot(rows, cols, index + 1)
        plt.plot(col_contents, dataframe[col], 'o', ms=1)
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        plt.title(col, fontsize=12, y=1)

    plt.tight_layout()

    if savepath is None:
        plt.show()

    else:
        plt.savefig(savepath)
        plt.close()


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.dir):
        os.makedirs(snakemake.output.dir)

    morph = pd.read_csv(snakemake.input.morph, sep='\t')
    genes = pd.read_csv(snakemake.input.gene, sep='\t')
    morph = morph.drop(columns=['label_id_cell', 'label_id_nucleus'])
    genes = genes.drop(columns=['label_id_cell', 'label_id_nucleus'])

    # done as here - https://stackoverflow.com/questions/38422893/correlation-matrix-of-one-dataframe-with-another
    merged = pd.concat([morph, genes], axis=1)
    correlation_p = merged.corr(method='pearson').filter(genes.columns).filter(morph.columns, axis=0)
    correlation_s = merged.corr(method='spearman').filter(genes.columns).filter(morph.columns, axis=0)

    path = os.path.join(snakemake.output.dir, 'pearson_correlation_all.pdf')
    correlation_plot(correlation_p, path)
    path = os.path.join(snakemake.output.dir, 'spearman_correlation_all.pdf')
    correlation_plot(correlation_s, path)

    # plot for all vs all scatter plots
    morph_list = [morph[var] for var in morph.columns]
    with ProcessPoolExecutor() as e:
        results = list(
            e.map(parallel_all_cols, [genes] * morph.shape[1], [snakemake.output.dir] * morph.shape[1],
                  morph.columns, morph_list))


if __name__ == '__main__':
    main()
