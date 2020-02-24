import pandas as pd
import numpy as np
import logging
import os
import pickle


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    summary = []
    # paths to pickled lists of each clustering
    cluster_paths = snakemake.input.clusters

    for path in cluster_paths:
        row = []

        # get k and resolution from filename & append to row
        file = os.path.split(path)[-1]
        file = os.path.splitext(file)[0]
        parts = file.split('_')
        row.append(parts[3])
        row.append(parts[5])

        with open(path, 'rb') as f:
            clus = pickle.load(f)

        # number of unique clusters
        no_unique = len(np.unique(clus))
        row.append(no_unique)

        # add clustering itself as rest of row
        row.extend(clus)
        summary.append(row)

    summary_tab = pd.DataFrame.from_records(summary)
    cols = ['k', 'res', 'no_clusters']
    cols.extend(list(range(0, len(clus))))
    summary_tab.columns = cols

    # summary table now contains one row per clustering with a column for the used k, the used res, the total number
    # of clusters, then the assignments for each row in the original table labelled from 0 to nrow(table)
    summary_tab.to_csv(snakemake.output.summary_table, index=False, sep='\t')


if __name__ == '__main__':
    main()
