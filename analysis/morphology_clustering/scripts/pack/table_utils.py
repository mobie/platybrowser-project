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

    Args:
        root_table [str] - path to root table (table with xyz positions etc tied to label_id)
            (can also pass pd.DataFrame directly)
        stats_table [str] - path to stats table (table with stats tied to label_id)
            (can also pass pd.DataFrame directly)
        out_path [str] - path to save result to (.csv)
        explore [bool] - whether to add columns for Explore Object Tables
        key [str] - name of column of stats table to use in merge
        intensity [str] - path to intensity image xml (optional - only needed for explore object tables)
        label [str] - path to label image xml (optional - only needed for explore object tables)
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


def generate_chromatin_table_from_stats(root_nuclei_table, stats_table, out_path, nuc_key='label_id',
                                        explore=False, intensity=None, label=None):
    """
    Generate a table with the chromatin labels from a table of stats based on nuclei. This is mostly so I can
    visualise stats on the chromatin segmentation. It will only keep nuclei present in the stats table.

    Args:
        root_nuclei_table [str] - path to root nucleus table
        stats_table [str] - path to table of nuclei stats (can also pass pd.DataFrame directly)
        nuc_key [str] - column name from stats_table to join on, should be the one containing nucleus ids
        out_path [str] - path to result location (.csv)
        explore [bool] - whether to add columns for Explore Object Tables
        intensity [str] - path to intensity image xml (optional - only needed for explore object tables)
        label [str] - path to label image xml (optional - only needed for explore object tables)
    """

    logger.info('Merging table of statistics to root & chromatin')
    logger.info('INPUT FILE - root table: %s', root_nuclei_table)
    logger.info('INPUT FILE - stats table: %s', stats_table)
    logger.info('INPUT FILE - intensity image: %s', intensity)
    logger.info('INPUT FILE - label image: %s', label)
    logger.info('OUTPUT FILE - merged table: %s', out_path)
    logger.info('PARAMETER - explore: %s', explore)

    root = pd.read_csv(root_nuclei_table, sep='\t')

    if type(stats_table) is str:
        stats = pd.read_csv(stats_table, sep='\t')
    else:
        stats = stats_table

    # merge root table into stats
    merged = stats.join(root.set_index('label_id'), on=nuc_key, how='left')
    merged2 = merged.copy()

    # add chromatin segmentation label ids
    merged['label_id_chromatin'] = merged[nuc_key]
    merged2['label_id_chromatin'] = merged[nuc_key] + 12000

    result = pd.concat([merged, merged2], axis=0)

    if explore:
        result['Path_IntensityImage'] = intensity
        result['Path_LabelImage'] = label

    result.to_csv(out_path, index=False, sep='\t')


def make_binary_columns(table, column, out_path):
    """
    Make binary columns (true - presence, false - absence) for all unique values in a certain column.
    In general useful for clustering columns, to expand this / make it easy to visualise individual clusters
    on Explore Object Tables

    Args:
        table [str] - path to table (can also pass pd.DataFrame directly)
        column [str] - name of column you want to expand
        out_path [str] - file path to save result (.csv)
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

    for val in unique_vals:
        new_col_name = str(val) + '_' + column
        table[new_col_name] = col == val

    table.to_csv(out_path, index=False, sep='\t')
