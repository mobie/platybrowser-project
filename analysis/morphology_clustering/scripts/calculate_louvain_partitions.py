import logging
import networkx as nx
import community
import pickle


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    KNN = nx.read_gpickle(snakemake.input.KNN)
    res = float(snakemake.params.res)

    # partition is a dictionary where the keys are the nodeids (i.e. the index of the rows from 0 to the full
    # length) & the values are the partition each is assigned to
    partition = community.best_partition(KNN, resolution=res, random_state=15)

    # change result into a list of the values ordered by the nodeid
    partition_listed = [partition[node] for node in sorted(partition.keys())]

    with open(snakemake.output.clusterings, 'wb') as f:
        pickle.dump(partition_listed, f)


if __name__ == '__main__':
    main()
