import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pack.Clustering import plot_clusters_on_umap


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    umap = np.loadtxt(snakemake.input.umap, delimiter='\t')

    k = int(snakemake.params.k)
    res = float(snakemake.params.res)

    with open(snakemake.input.cluster, 'rb') as f:
        clustering = pickle.load(f)

    plot, no_clusters, no_singletons = plot_clusters_on_umap(umap, clustering, label=True)
    plot.set_title(
        'k_%s_res_%s__no_clusters_without_singletons__%s__no_singletons__%s' % (
        k, res, no_clusters - no_singletons, no_singletons), fontsize=20)

    plt.savefig(snakemake.output.fig)


if __name__ == '__main__':
    main()
