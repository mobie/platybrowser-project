import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def merge_root_stats_tables(root_table, stats_table, out_path, key='label_id', explore=False, intensity=None,
                            label=None):
    """
    Merge a root table (this is the main table with xyz positions etc tied to label_id) into a stats table.
    Will keep all rows in stats and add columns from the root (won't add rows from root not included in stats)
    Optionally add the columns needed for Explore Objects Tables

    :param root_table: string, path to root table (table with xyz positions etc tied to label_id)
    :param stats_table: string, path to stats table (table with stats tied to label_id)
    :param out_path: string, path to save result to
    :param explore: boolean, whether to add columns for explore object tables
    :param key: key (column name) of stats table to use in merge
    :param intensity: string, path to intensity image xml (optional)
    :param label: string, path to label image xml (optional)
    """
    logger.info('Merging table of statistics with root table')
    logger.info('INPUT FILE - root table: %s', root_table)
    logger.info('INPUT FILE - stats table: %s', stats_table)
    logger.info('INPUT FILE - intensity image: %s', intensity)
    logger.info('INPUT FILE - label image: %s', label)
    logger.info('OUTPUT FILE - merged table: %s', out_path)
    logger.info('PARAMETER - explore: %s', explore)

    if type(root_table) is str:
        root = pd.read_csv(root_table, sep='\t')
    else:
        root = root_table

    if type(stats_table) is str:
        stats = pd.read_csv(stats_table, sep='\t')
    else:
        stats = stats_table

    merged = stats.join(root.set_index('label_id'), on=key, how='left', lsuffix='_root', rsuffix='_stats')

    if explore:
        merged['Path_IntensityImage'] = intensity
        merged['Path_LabelImage'] = label

    merged.to_csv(out_path, index=False, sep='\t')


def make_binary_columns(table, column, out_path):
    """
    Make binary columns (true - presence, false - absence) for all unique values in a certain column.
    In general useful for clustering columns, to expand this / make it easy to visualise individual clusters
    on Explore Object Tables
    :param table: string, path to table, or pandas dataframe
    :param column: string, name of column you want to expand
    :param out_path: string, file path to save
    """

    logger.info("Creating table for visualisation - added binary columns")
    if type(table) == str:
        logger.info("INPUT FILE - table %s", table)
    logger.info("PARAMETER - column %s", column)
    logger.info("OUTPUT FILE - out_path %s", out_path)

    if type(table) == str:
        table = pd.read_csv(table, sep='\t')

    col = table[column]

    unique_vals = np.unique(col)

    # singletons
    singletons = np.zeros(col.shape)

    for val in unique_vals:
        new_col_name = str(val) + '_' + column
        new_col = col == val
        # if it's a singleton column
        if np.sum(new_col) == 1:
            singletons[new_col] = 1
        else:
            table[new_col_name] = col == val

    if np.sum(singletons) != 0:
        table['singletons'] = singletons

    table.to_csv(out_path, index=False, sep='\t')
