import pandas as pd
import logging
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    table = pd.read_csv(snakemake.input.table, delimiter='\t')
    # drop label id
    for var in ['label_id', 'unique_id']:
        if var in table.columns:
            table = table.drop(columns=[var])

    k = int(snakemake.params.k)
    metric = snakemake.params.metric

    # Calculate KNN and convert to networkx format
    # mode = 'distance' for weighted edges and 'connectivity' for just 0/1 edges
    KNN = kneighbors_graph(table, n_neighbors=k, mode='distance', metric=metric)
    net_graph = nx.from_scipy_sparse_matrix(KNN)

    nx.write_gpickle(net_graph, snakemake.output.KNN)


if __name__ == '__main__':
    main()
