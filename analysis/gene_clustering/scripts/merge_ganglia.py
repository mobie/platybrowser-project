import pandas as pd
import numpy as np
import logging


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # only keep column of merged bilateral ids - rename to ganglion_id
    ganglia = pd.read_csv(snakemake.input.ganglia, sep='\t')
    ganglia = ganglia[['label_id', 'merged_bilateral_ganglion_id']]
    ganglia.columns = ['label_id', 'ganglion_id']

    # drop nans
    ganglia = ganglia.dropna()

    ganglia.to_csv(snakemake.output.ganglia_merged, index=False, sep='\t')


if __name__ == '__main__':
    main()
