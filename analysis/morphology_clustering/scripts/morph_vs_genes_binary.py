import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from concurrent.futures import ProcessPoolExecutor


# wrapper to call column_against_rest_plot & set up required filepaths to run in parallel
def parallel_all_cols(dataframe, path_root, col_name, col_contents, stats):
    path = os.path.join(path_root, 'scatter_plots_%s_vs_all.png' % col_name)
    column_against_rest_plot(dataframe, col_contents, stats, path)


def column_against_rest_plot(dataframe, col_contents, stats, savepath=None):
    """
        Produces violin plots for every column in dataframe vs the provided col_contents
        Col_contents on y, dataframe columns on x. Add wilcox text values also.

        Args:
            dataframe [pd.Dataframe] - a dataframe to plot
            col_contents [listlike] - contents of a column
            stats [tuple] - row from dataframe of wilcox stats
            savepath [str] - where to save the resulting image (.png), None to just show it
        """
    # set up number of cols / rows needed for grid of violin plots
    sqrt_cols = np.sqrt(dataframe.shape[1])
    cols = int(sqrt_cols)
    remaining = dataframe.shape[1] - cols
    rows = 1 + int(np.ceil(remaining / cols))

    plt.figure(figsize=(30, 30))
    sns.set()
    sns.set_style('white')
    for index, col in enumerate(dataframe.columns):
        x = plt.subplot(rows, cols, index + 1)
        sns.violinplot(x=dataframe[col], y=col_contents)
        wilcox_corrected = round(stats[index], 5)
        plt.title('%s--%s' % (col, wilcox_corrected), fontsize=12, y=1)
        x.set_xlabel('')
        x.set_ylabel('')

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

    if not os.path.isdir(snakemake.output.dir):
        os.makedirs(snakemake.output.dir)

    morph = pd.read_csv(snakemake.input.morph, sep='\t')
    genes = pd.read_csv(snakemake.input.gene, sep='\t')
    wilcox_corrected = pd.read_csv(snakemake.input.wilcox_corrected, sep='\t')

    morph = morph.drop(columns=['label_id_cell', 'label_id_nucleus'])
    genes = genes.drop(columns=['label_id_cell', 'label_id_nucleus'])

    # plot for all vs all scatter plots
    morph_list = [morph[var] for var in morph.columns]
    # here use ordinary tuple as named tuples can't be pickled
    stats_list = [stat for stat in wilcox_corrected.itertuples(index=False, name=None)]
    with ProcessPoolExecutor() as e:
        results = list(
            e.map(parallel_all_cols, [genes] * morph.shape[1], [snakemake.output.dir] * morph.shape[1],
                  morph.columns, morph_list, stats_list))


if __name__ == '__main__':
    main()
