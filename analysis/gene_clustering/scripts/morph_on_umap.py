import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pack.Clustering import plot_continuous_feature_on_umap


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.morph_dir):
        os.makedirs(snakemake.output.morph_dir)

    umap = np.loadtxt(snakemake.input.umap, delimiter='\t')
    table = pd.read_csv(snakemake.input.table, sep='\t')
    morph = pd.read_csv(snakemake.input.merged_morph, sep='\t')

    if 'unique_id' in table.columns:
        raise ValueError(
            'Unique_id column in table - morphology can not be plotted for tables with duplicates removed. Each'
            'row may represent multiple cells with identical gene expression & hence be a mix of different'
            'morphologies')

    table = table.loc[:, table.columns == 'label_id']
    table = table.join(morph.set_index('label_id'), on='label_id', how='left')
    table = table.drop(columns=['label_id'])
    # remove any nans - morphology tables filter for failures of certain statistics so may
    # lead to na rows in table, here just replace with 0
    table = table.fillna(value=0)

    for val in list(table.columns):
        plot = plot_continuous_feature_on_umap(umap, table[val])
        plot.set_title(val, fontsize=20)
        path = os.path.join(snakemake.output.morph_dir, val + '.png')
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    main()
