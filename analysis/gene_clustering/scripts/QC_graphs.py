import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging


def correlation_plot(dataframe, save_path=None):
    """
    All by all (columns) correlation heatmap
    :param dataframe: pandas dataframe
    :param save_path: Path to save result (.pdf), None to just show result
    """
    sns.set(font_scale=0.5)
    sns.set_style('ticks')
    sns.clustermap(dataframe, cmap='mako', figsize=(30, 30))

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)
        plt.close()


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.QC):
        os.makedirs(snakemake.output.QC)

    stats_cut = pd.read_csv(snakemake.input.QC_stats, sep='\t')

    # remove label ids if present
    for var in ['label_id', 'unique_id']:
        if var in stats_cut.columns:
            stats_cut = stats_cut.drop(columns=[var])

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


if __name__ == '__main__':
    main()
