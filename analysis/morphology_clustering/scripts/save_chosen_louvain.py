import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pack.Clustering import plot_clusters_on_umap
from pack.table_utils import make_binary_columns


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    umap = np.loadtxt(snakemake.input.umap, delimiter='\t')

    chosen_k = int(snakemake.params.chosen_k)
    chosen_res = float(snakemake.params.chosen_res)

    if chosen_k == 0:
        raise ValueError('No chosen number of neighbours - k - for louvain')

    elif chosen_res == 0:
        raise ValueError('No chosen resolution for louvain')

    with open(snakemake.input.cluster_file, 'rb') as f:
        clustering = pickle.load(f)

    plot, no_clusters, no_singletons = plot_clusters_on_umap(umap, clustering, label=True)
    plot.set_title(
        'k_%s_res_%s__no_clusters__%s__no_singletons__%s' % (chosen_k, chosen_res, no_clusters, no_singletons),
        fontsize=20)

    plt.savefig(snakemake.output.fig)

    table = pd.read_csv(snakemake.input.filtered, sep='\t')

    col_name = 'chosen_k_%s_res_%s' % (chosen_k, chosen_res)
    table[col_name] = clustering

    # just keep label id and clusters
    table = table[['label_id_nucleus', 'label_id_cell', col_name]]
    table.to_csv(snakemake.output.merged_table, index=False, sep='\t')

    # make viz table nuc
    viz_table = pd.read_csv(snakemake.input.viz_table_nuc, sep='\t')
    viz_table['clusters'] = clustering
    make_binary_columns(viz_table, 'clusters', snakemake.output.viz_table_nuc)

    # make viz table cells
    viz_table = pd.read_csv(snakemake.input.viz_table_cell, sep='\t')
    viz_table['clusters'] = clustering
    make_binary_columns(viz_table, 'clusters', snakemake.output.viz_table_cell)


if __name__ == '__main__':
    main()
