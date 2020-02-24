import pandas as pd
import numpy as np
import logging
import pack.table_utils as table_utils
import pack.utils as utils


def filter_table_size(table, min_size, max_size):
    """ Filter table based on min and max no of pixels in an object """
    if max_size is None:
        table = table.loc[table['n_pixels'] >= min_size, :]
    else:
        criteria = np.logical_and(table['n_pixels'] > min_size, table['n_pixels'] < max_size)
        table = table.loc[criteria, :]
    return table


# some cell segmentations have a sensible total pixel size but very large bounding boxes i.e. they are small spots
# distributed over a large region & not one cell > we must filter out these cases by capping the bounding box size
def filter_table_bb(table, max_bb):
    """ Filter table - retain only those with bounding box volume < max_bb """
    total_bb = (table['bb_max_z'] - table['bb_min_z']) * (table['bb_max_y'] - table['bb_min_y']) * (
            table['bb_max_x'] - table['bb_min_x'])

    table = table.loc[total_bb < max_bb, :]

    return table


def filter_table_from_mapping(table, mapping_path):
    """ Retain only cells that have a corresponding nucleus """

    # first column cell id, second nucleus id
    mapping = pd.read_csv(mapping_path, sep='\t')

    # remove zero labels from this table, if exist
    mapping = mapping.loc[np.logical_and(mapping.iloc[:, 0] != 0,
                                         mapping.iloc[:, 1] != 0), :]
    table = table.loc[np.isin(table['label_id'], mapping['label_id']), :]

    return table


def filter_table_region(table, region_path, regions=('empty', 'yolk', 'neuropil', 'cuticle'), exclude=True):
    """ Filter cells that lie in particular regions - exclude these regions if exclude == True, otherwise just
    retain cells in those regions """

    region_mapping = pd.read_csv(region_path, sep='\t')

    # remove zero label if it exists
    region_mapping = region_mapping.loc[region_mapping['label_id'] != 0, :]

    if exclude:
        for region in regions:
            region_mapping = region_mapping.loc[region_mapping[region] == 0, :]

    else:
        cut_table = region_mapping[regions]
        criteria = (cut_table == 1).any(axis=1)
        region_mapping = region_mapping.loc[criteria, :]

    table = table.loc[np.isin(table['label_id'], region_mapping['label_id']), :]

    return table


def filter_table_extrapolated_intensity(table, extrapolated_intensity_path):
    """ Exclude cells in the region where the intensity correction was extrapolated """

    extrapolated_intensity = pd.read_csv(extrapolated_intensity_path, sep='\t')
    extrapolated_intensity = extrapolated_intensity.loc[extrapolated_intensity.has_extrapolated_intensities == 0, :]

    table = table.loc[np.isin(table['label_id'], extrapolated_intensity['label_id']), :]
    return table


def filter_no_expression(table):
    """ Remove any genes with zero expression """

    # Remove any columns (genes) with no expression in any cell
    table = table.loc[:, (table != 0).any(axis=0)]

    # remove any rows (cells) with zero expression
    table_cut = table.copy()
    for var in ['label_id', 'unique_id']:
        if var in table_cut.columns:
            table_cut = table_cut.drop(columns=[var])

    criteria = (table_cut != 0).any(axis=1)
    table = table.loc[criteria, :]

    return table


def remove_genes(table, genes_to_exclude):
    """ Remove specified genes - usually ones that are stains etc... """

    for var in genes_to_exclude:
        if var in table.columns:
            table = table.drop(columns=[var])

    return table


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Read in gene data
    genes = pd.read_csv(snakemake.input.genes, sep='\t')
    logger.info('Read in gene table with %s rows' % genes.shape[0])

    # remove nas & unwanted gene columns
    genes = genes.dropna(axis=0, how='any')
    genes = remove_genes(genes, snakemake.params.genes_to_exclude)
    logger.info('%s rows remain after filtering missing values and excluded genes' % genes.shape[0])

    # Save copy with columns for Explore Object Tables plugin
    genes_copy = genes.copy()
    table_utils.merge_root_stats_tables(snakemake.input.cell_root, genes_copy, snakemake.output.gene_nonans_viz,
                                        key='label_id',
                                        explore=True, intensity=snakemake.input.intensity,
                                        label=snakemake.input.cell_seg)

    # Do same QC here as I would for calculating cell morphology stats - remove those with no assigned nucleus,
    # remove the ones that are too small / big, remove ones in certain regions
    cell_root = pd.read_csv(snakemake.input.cell_root, sep='\t')
    cell_root = filter_table_from_mapping(cell_root, snakemake.input.cell_nuc_mapping)
    cell_root = filter_table_region(cell_root, snakemake.input.cell_region_mapping)
    cell_root = filter_table_size(cell_root, snakemake.params.min_size_cell_px, snakemake.params.max_size_cell_px)
    cell_root = filter_table_bb(cell_root, snakemake.params.max_bounding_box_size_cell)

    # optionally remove cells that are in the region for extrapolated intensities
    if snakemake.params.extrapolated_intensity_cell != '':
        cell_root = filter_table_extrapolated_intensity(cell_root, snakemake.input.extrapolated_intensity_cell)

    # optionally only keep cells in a certain region
    if snakemake.params.filter_regions != []:
        cell_root = filter_table_region(cell_root, snakemake.input.cell_region_mapping, snakemake.params.filter_regions,
                                        exclude=False)

    genes = genes.loc[np.isin(genes['label_id'], cell_root['label_id']), :]

    # remove any genes / cells with zero expression
    genes = filter_no_expression(genes)

    logger.info('%s rows remain after QC' % genes.shape[0])

    # save QCd stats as is, then with columns for Explore Object Tables
    genes.to_csv(snakemake.output.after_QC, index=False, sep='\t')
    genes_copy = genes.copy()
    table_utils.merge_root_stats_tables(snakemake.input.cell_root, genes_copy,
                                        snakemake.output.after_QC_viz, key='label_id',
                                        explore=True,
                                        intensity=snakemake.input.intensity,
                                        label=snakemake.input.cell_seg)

    # save binary
    just_genes = genes.drop(columns=['label_id'])
    just_genes = (just_genes > snakemake.params.gene_threshold).astype(int)
    just_genes.insert(0, 'label_id', genes['label_id'])
    # filter rows/columns with no expression under new threshold
    just_genes = filter_no_expression(just_genes)

    logger.info('%s rows remain after binary QC' % just_genes.shape[0])

    # save QCd binary stats as is, then with columns for Explore Object Tables
    just_genes.to_csv(snakemake.output.after_QC_binary, index=False, sep='\t')

    genes_copy = just_genes.copy()
    table_utils.merge_root_stats_tables(snakemake.input.cell_root, genes_copy,
                                        snakemake.output.after_QC_binary_viz, key='label_id',
                                        explore=True,
                                        intensity=snakemake.input.intensity,
                                        label=snakemake.input.cell_seg)


if __name__ == '__main__':
    main()
