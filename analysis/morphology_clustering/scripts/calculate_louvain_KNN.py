import numpy as np
import logging
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    table = np.loadtxt(snakemake.input.table, delimiter='\t')
    k = int(snakemake.params.k)

    # Calculate KNN and convert to networkx format
    KNN = kneighbors_graph(table, n_neighbors=k, mode='distance')
    net_graph = nx.from_scipy_sparse_matrix(KNN)

    nx.write_gpickle(net_graph, snakemake.output.KNN)


if __name__ == '__main__':
    main()
