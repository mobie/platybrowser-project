import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import pack.table_utils as table_utils

log_path = snakemake.log[0]
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_table_extrapolated_intensity(table, extrapolated_intensity_path):
    """ Exclude cells in the region where the intensity correction was extrapolated """

    extrapolated_intensity = pd.read_csv(extrapolated_intensity_path, sep='\t')
    extrapolated_intensity = extrapolated_intensity.loc[extrapolated_intensity.has_extrapolated_intensities == 0, :]

    table = table.loc[np.isin(table['label_id'], extrapolated_intensity['label_id']), :]
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


def filter_all_zero_cols(table):
    """ Remove columns that are all zero - probably failed computation of statistic """
    criteria = (table == 0).all()
    logger.info('Removed columns as all zeros ' + str(table.columns[criteria]))
    return table.loc[:, np.invert(criteria)]


def filter_texture_fails(table):
    """ Filter out rows that fail any of the haralick texture statistics. This often happens with the fully dark nuclei
    as euchromatin is only segmented as an erroneous rim around the outside. The small number of pixels and odd shape
    of this rim means haralick feature computations fail """

    texture = table.filter(regex='texture')
    criteria = (texture == 0).any(axis=1)

    table = table.loc[np.invert(criteria), :]
    return table


def main():
    cell_stats = pd.read_csv(snakemake.input.morph_stats_cells, sep='\t')
    logger.info('Read in cell stats with %s rows' % cell_stats.shape[0])
    cell_stats = cell_stats.dropna(axis=0, how='any')
    logger.info('%s rows remain after filtering missing values' % cell_stats.shape[0])

    # Save copy with columns for Explore Object Tables plugin
    stats_copy = cell_stats.copy()
    table_utils.merge_root_stats_tables(snakemake.input.cell_root, stats_copy, snakemake.output.cell_nonans_viz,
                                        key='label_id',
                                        explore=True, intensity=snakemake.input.intensity,
                                        label=snakemake.input.cell_seg)

    # Don't need to filter for size / presence of matching nucleus / certain regions etc here as this is done when
    # the morphology stats are calculated
    # Only filter to remove those in the extrapolated intensity region & any all zero columns &
    # optionally any rows that fail haralick computations
    cell_stats = filter_table_extrapolated_intensity(cell_stats, snakemake.input.extrapolated_intensity_cell)
    cell_stats = filter_all_zero_cols(cell_stats)
    if snakemake.params.texture_fails == 'True':
        cell_stats = filter_texture_fails(cell_stats)

    if snakemake.params.filter_region != []:
        cell_stats = filter_table_region(cell_stats, snakemake.input.cell_region_mapping,
                                         snakemake.params.filter_region, exclude=False)
    logger.info('%s rows remain after QC' % cell_stats.shape[0])
    cell_stats.columns = ['%s_cell' % col for col in cell_stats.columns]

    # Nuclei morphology stats
    nuclei_stats = pd.read_csv(snakemake.input.morph_stats_nuclei, sep='\t')
    logger.info('Read in nuclei stats with %s rows' % nuclei_stats.shape[0])
    nuclei_stats = nuclei_stats.dropna(axis=0, how='any')
    logger.info('%s rows remain after filtering missing values' % nuclei_stats.shape[0])

    # Save copy with columns for Explore Object Tables plugin
    stats_copy = nuclei_stats.copy()
    table_utils.merge_root_stats_tables(snakemake.input.nuclei_root, stats_copy, snakemake.output.nuclei_nonans_viz,
                                        key='label_id',
                                        explore=True, intensity=snakemake.input.intensity,
                                        label=snakemake.input.nuclei_seg)

    nuclei_stats = filter_table_extrapolated_intensity(nuclei_stats, snakemake.input.extrapolated_intensity_nuc)
    nuclei_stats = filter_all_zero_cols(nuclei_stats)
    if snakemake.params.texture_fails == 'True':
        nuclei_stats = filter_texture_fails(nuclei_stats)
    logger.info('%s rows remain after QC' % nuclei_stats.shape[0])
    nuclei_stats.columns = ['%s_nucleus' % col for col in nuclei_stats.columns]

    # Merge tables - only keeping cells with nuclei in nuc table & vice versa, only nuclei with cells in cell table
    # Will lose some nuclei that aren't assigned to a cell but, as we want to look at gene expression, this is
    # necessary as gene expression is only assigned at the level of cells, not at the level of individual nuclei

    # first column cell id, second nucleus id
    cell_nuc_mapping = pd.read_csv(snakemake.input.cell_nuc_mapping, sep='\t')
    cell_nuc_mapping = cell_nuc_mapping.loc[np.logical_and(cell_nuc_mapping.iloc[:, 0] != 0,
                                                           cell_nuc_mapping.iloc[:, 1] != 0), :]
    # check each nucleus only appears once
    counts_nuc = np.unique(cell_nuc_mapping['nucleus_id'], return_counts=True)[1]
    counts_cell = np.unique(cell_nuc_mapping['label_id'], return_counts=True)[1]
    if sum(counts_nuc > 1) > 0 or sum(counts_cell > 1) > 0:
        logger.info('Nuclei or cells appear more than once')
        raise ValueError('Nuclei or cells appear more than once in cell-nucleus mapping!')

    else:
        cell_nuc_mapping = cell_nuc_mapping.rename(columns={"label_id": "label_id_cell"})
        stats = nuclei_stats.join(cell_nuc_mapping.set_index('nucleus_id'), on='label_id_nucleus', how='inner')
        stats = stats.join(cell_stats.set_index('label_id_cell'), on='label_id_cell', how='inner')
        logger.info('%s missing values in cell column' % sum(np.isnan(stats['label_id_cell'])))
        logger.info('%s rows remain after joining cells and nuclei, only keeping one to one matches' % stats.shape[0])

    # save QCd stats with no extra columns
    stats.to_csv(snakemake.output.after_QC, index=False, sep='\t')

    # save QCd stats with viz columns
    stats_copy = stats.copy()
    table_utils.merge_root_stats_tables(snakemake.input.nuclei_root, stats_copy,
                                        snakemake.output.after_QC_nuc_viz, key='label_id_nucleus',
                                        explore=True,
                                        intensity=snakemake.input.intensity,
                                        label=snakemake.input.nuclei_seg)

    table_utils.merge_root_stats_tables(snakemake.input.cell_root, stats_copy,
                                        snakemake.output.after_QC_cell_viz, key='label_id_cell',
                                        explore=True, intensity=snakemake.input.intensity,
                                        label=snakemake.input.cell_seg)

    table_utils.generate_chromatin_table_from_stats(snakemake.input.nuclei_root, stats_copy,
                                                    snakemake.output.after_QC_chromatin_viz,
                                                    nuc_key='label_id_nucleus',
                                                    explore=True,
                                                    intensity=snakemake.input.intensity,
                                                    label=snakemake.input.chromatin_seg)

    labels = ['label_id_nucleus', 'label_id_cell']
    just_label = stats[labels]

    # xyz coordinate tables for QCd cells/nuclei
    transformed_xyz = pd.read_csv(snakemake.input.transformed_nuc_coords, sep='\t')
    midline_dist = pd.read_csv(snakemake.input.midline_distance_nuc, sep='\t')
    transformed_xyz_cut = transformed_xyz[['label_id', 'new_x', 'new_y', 'new_z']]
    midline_cut = midline_dist[['label_id', 'side', 'side2', 'absolute']]
    merged_xyz = just_label.join(transformed_xyz_cut.set_index('label_id'), on='label_id_nucleus', how='left')
    merged_xyz = merged_xyz.join(midline_cut.set_index('label_id'), on='label_id_nucleus', how='left')
    merged_xyz.to_csv(snakemake.output.QC_xyz, index=False, sep='\t')

    # genes overlap
    genes_overlap = pd.read_csv(snakemake.input.genes_overlap, sep='\t')
    logger.info('Read in gene table with %s rows' % genes_overlap.shape[0])

    just_label = stats[labels]
    merged_genes = just_label.join(genes_overlap.set_index('label_id'), on='label_id_cell', how='left')
    merged_genes.to_csv(snakemake.output.QC_genes_overlap, index=False, sep='\t')

    # binarised genes overlap
    just_genes = merged_genes.drop(columns=labels)
    just_genes = (just_genes > snakemake.params.gene_overlap_threshold).astype(int)
    for val in labels:
        just_genes.insert(0, val, merged_genes[val])
    just_genes.to_csv(snakemake.output.QC_genes_overlap_binary, index=False, sep='\t')

    # genes VC
    genes_vc = pd.read_csv(snakemake.input.genes_vc, sep='\t')
    merged_genes = just_label.join(genes_vc.set_index('label_id'), on='label_id_cell', how='left')
    merged_genes.to_csv(snakemake.output.QC_genes_vc, index=False, sep='\t')

    # binarised genes vc
    just_genes = merged_genes.drop(columns=labels)
    just_genes = (just_genes > snakemake.params.gene_vc_threshold).astype(int)
    for val in labels:
        just_genes.insert(0, val, merged_genes[val])
    just_genes.to_csv(snakemake.output.QC_genes_vc_binary, index=False, sep='\t')

    # genes region
    cell_region_mapping = pd.read_csv(snakemake.input.cell_region_mapping, sep='\t')
    merged_regions = just_label.join(cell_region_mapping.set_index('label_id'), on='label_id_cell', how='left')
    merged_regions.to_csv(snakemake.output.QC_region, index=False, sep='\t')


if __name__ == '__main__':
    main()
