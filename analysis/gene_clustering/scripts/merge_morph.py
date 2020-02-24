import pandas as pd
import numpy as np
import logging

log_path = snakemake.log[0]
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_all_zero_cols(table):
    """ Remove columns that are all zero - probably failed computation of statistic """
    criteria = (table == 0).all()
    logger.info('Removed columns as all zeros ' + str(table.columns[criteria]))
    return table.loc[:, np.invert(criteria)]


def main():
    morph_cells = pd.read_csv(snakemake.input.morph_cells, sep='\t')
    morph_nuclei = pd.read_csv(snakemake.input.morph_nuclei, sep='\t')
    cell_nuc_mapping = pd.read_csv(snakemake.input.cell_nuc_mapping, sep='\t')

    morph_cells = filter_all_zero_cols(morph_cells)
    morph_nuclei = filter_all_zero_cols(morph_nuclei)

    morph_cells.columns = ['%s_cell' % col for col in morph_cells.columns]
    morph_nuclei.columns = ['%s_nucleus' % col for col in morph_nuclei.columns]

    # columns label_id and nucleus_id
    stats = morph_cells.join(cell_nuc_mapping.set_index('label_id'), on='label_id_cell', how='left')
    stats = stats.join(morph_nuclei.set_index('label_id_nucleus'), on='nucleus_id', how='left')

    stats = stats.drop(columns=['nucleus_id'])
    stats = stats.rename(columns={"label_id_cell": "label_id"})

    stats.to_csv(snakemake.output.merged_morph, index=False, sep='\t')


if __name__ == '__main__':
    main()
