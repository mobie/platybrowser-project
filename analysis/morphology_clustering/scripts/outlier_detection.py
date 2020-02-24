import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import logging


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stats_table = pd.read_csv(snakemake.input.morph, sep='\t', header=None)

    clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    pred = clf.fit_predict(stats_table)
    scores = clf.negative_outlier_factor_

    # >1 is outlier
    # - 1 for outlier

    # write out result
    viz_table = pd.read_csv(snakemake.input.viz_table_nuc, sep='\t')
    viz_table['novel'] = pred == -1
    viz_table.to_csv(snakemake.output.table_nuc, index=False, sep='\t')

    viz_table = pd.read_csv(snakemake.input.viz_table_cell, sep='\t')
    viz_table['novel'] = pred == -1
    viz_table.to_csv(snakemake.output.table_cell, index=False, sep='\t')


if __name__ == '__main__':
    main()
