import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import multiprocessing
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
import pickle


#  returns INDEX of result, not label
def neighbours(table):
    """ All by all neighbour ranking, returns table of rows == rows of input table, columns being indices of
    neighbours. e.g. column 1 is index of the first neighbour, column 2 is index of the second neighbour and so on..."""
    nbrs = NearestNeighbors(n_neighbors=table.shape[0], algorithm='ball_tree', metric='euclidean',
                            n_jobs=multiprocessing.cpu_count()).fit(table)
    distances, indices = nbrs.kneighbors(table, return_distance=True)

    # first column is the nucleus itself, so exclude this
    result = indices[:, 1:]

    return result


def all_by_all_xyz_criteria(transformed_xyz, midline_dist, bilateral_dist):
    """ amalgamate tables to give one table that gives True/False depending on if all xyz criteria are met between
    pairs of cells (nrow == nrow in morphology table, one for each cell. Same number of columns so that e.g.
    location [9, 12] will be True if the cell at index 9 and the cell at index 12 meet the xyz criteria

    xyz criteria-
     transformed y/z coordinate (prospr space) within mean+-2 standard deviations
     absolute distance from midline within mean +- 2 standard deviations
     on opposite sides of the midline
     """
    y_vals = transformed_xyz.new_y[:, None]
    z_vals = transformed_xyz.new_z[:, None]
    abs_mid = midline_dist.absolute[:, None]
    side = midline_dist.side.astype(int)[:, None]

    # all vs all absolute euclidean distance
    # binarise based on bilateral_dist stats
    distances_y = np.abs(squareform(pdist(y_vals)))
    distances_y = distances_y <= float(
        bilateral_dist.loc[bilateral_dist.stat == 'new_y', 'mean_with_two_sds'])

    distances_z = np.abs(squareform(pdist(z_vals)))
    distances_z = distances_z <= float(
        bilateral_dist.loc[bilateral_dist.stat == 'new_z', 'mean_with_two_sds'])

    distances_abs = np.abs(squareform(pdist(abs_mid)))
    distances_abs = distances_abs <= float(
        bilateral_dist.loc[bilateral_dist.stat == 'absolute', 'mean_with_two_sds'])

    # if cells are on opposite sides of the midline, then their distance will be 1 (i.e. side column 0,1 or 1,0)
    opp_sides = squareform(pdist(side))
    opp_sides = opp_sides == 1

    # combine all tables to give one with True/False as to whether fulfills all criteria
    all_by_all_criteria = np.logical_and(distances_y, distances_z)
    all_by_all_criteria = np.logical_and(all_by_all_criteria, distances_abs)
    all_by_all_criteria = np.logical_and(all_by_all_criteria, opp_sides)

    return all_by_all_criteria


def calculate_all_neighbours(morph_stats_cut, all_by_all_nucleus):
    """ Run all calculations for neighbours based on the provided morphology table, and the validation / xyz
    criteria"""

    normed_morph = StandardScaler().fit_transform(morph_stats_cut)
    normed_morph = pd.DataFrame(normed_morph)
    normed_morph.columns = morph_stats_cut.columns

    # all by all neighbours, rows == no of rows in normed_morph, one for each cell
    # columns give index of neighbours (1st column == 1st neighbour, 2nd column == 2nd neighbour and so on...)
    nn = neighbours(normed_morph)

    # result using nuclei xyz position criteria - list of len() equal to number of rows in morphology table,
    # each entry is the number of neighbours away the first cell to meet the xyz criteria occurs
    result_nuc_criteria = get_nearest_index(nn, all_by_all_nucleus)

    # Cells where no cells meet the xyz criteria have values of -1

    # check - each row of nn should contain all possible indices (0 - nrow of nn) apart from its own index
    unique_vals = np.array(range(0, nn.shape[0]))
    for i in range(0, nn.shape[0]):
        row = nn[i, :]
        test = np.sort(unique_vals[unique_vals != i])
        row_vals = np.sort(np.unique(row))
        if not (test == row_vals).all():
            print(i)
            print('row does not contain all values apart from its own index')

    # calculate randomised results
    nuc_results_random = []
    for i in range(0, 100):
        print(i)
        # shuffle each row & re-run
        nn_shuffle = nn.copy()
        for j in range(0, nn.shape[0]):
            np.random.shuffle(nn_shuffle[j, :])

        nuc_results_random.append(get_nearest_index(nn_shuffle, all_by_all_nucleus))

    # one row per iteration, number of cols == number of rows in morphology table. Each entry is the number of
    # neighoburs away the first cell to meet the xyz criteria occurs
    nuc_results_random = pd.DataFrame.from_records(nuc_results_random)

    return result_nuc_criteria, nuc_results_random


def get_nearest_index(nn, all_by_all_criteria):
    """ Return index of cell that is closest in morphology space (and fulfills the xyz criteria) for each row in nn"""

    # batches of 1000 rows to run in parallel
    row_ids = list(range(0, nn.shape[0]))
    batched_row_ids = []
    for i in range(0, len(row_ids), 1000):
        batched_row_ids.append(row_ids[i:i + 1000])

    with ProcessPoolExecutor() as e:
        results = list(
            e.map(get_nearest_index_row_batch, batched_row_ids, [nn] * len(batched_row_ids),
                  [all_by_all_criteria] * len(batched_row_ids)))

    flat_results = [item for sublist in results for item in sublist]

    return flat_results


def get_nearest_index_row_batch(row_ids, nn, all_by_all_valid):
    """ For each row id given, loop through each column of that row (i.e. the indices of each of its neighbours)
    & find the first one that meets the xyz criteria """

    result = []
    for row_id in row_ids:
        # give value of -1 if no cells that meet the xyz criteria are found
        val_to_add = -1
        for pair_id in range(0, nn.shape[1]):
            pair_index = nn[row_id, pair_id]

            # if meets xyz criteria append the index
            if all_by_all_valid[row_id, pair_index]:
                val_to_add = pair_id + 1
                break

        result.append(val_to_add)

    return result


def main():
    morph_stats = pd.read_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\1-0-1\\tables\\after_QC.csv',
        sep='\t')
    morph_stats_cut = morph_stats.drop(columns=['label_id_cell', 'label_id_nucleus'])

    transformed_xyz_nucleus = pd.read_csv(
        'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/prospr_space_nuclei_points_1_0_1.csv',
        sep='\t')
    transformed_xyz_nucleus = transformed_xyz_nucleus.rename(columns={'label_id': 'nucleus_id'})
    midline_dist_nucleus = pd.read_csv(
        'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/distance_from_midline_nuclei_1_0_1.csv',
        sep='\t')
    midline_dist_nucleus = midline_dist_nucleus.rename(columns={'label_id': 'nucleus_id'})

    bilateral_dist_nuc = pd.read_csv(
        'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/bilateral_distance_summary_nuclei_1_0_1.csv',
        sep='\t')

    final_labels_nucleus_df = morph_stats.loc[:, morph_stats.columns == 'label_id_nucleus']

    # tables for xyz validation - get label ids in same order as the morphology table
    transformed_xyz_nucleus = transformed_xyz_nucleus[['nucleus_id', 'new_y', 'new_z']]
    transformed_xyz_nucleus = final_labels_nucleus_df.join(transformed_xyz_nucleus.set_index('nucleus_id'),
                                                           on='label_id_nucleus',
                                                           how='left')
    midline_dist_nucleus = midline_dist_nucleus[['nucleus_id', 'side', 'absolute']]
    midline_dist_nucleus = final_labels_nucleus_df.join(midline_dist_nucleus.set_index('nucleus_id'),
                                                        on='label_id_nucleus',
                                                        how='left')
    # amalgamate tables to give one table that gives True/False depending on if all xyz criteria are met between
    # pairs of nuclei (nrow == nrow in morphology table, one for each cell. Same number of columns so that e.g.
    # location [9, 12] will be True if the cell at index 9 and the cell at index 12 meet the xyz criteria (similar y/z,
    # similar distance from the midline & on opposite sides of the midline)
    all_by_all_nucleus = all_by_all_xyz_criteria(transformed_xyz_nucleus, midline_dist_nucleus, bilateral_dist_nuc)

    # using all stats
    nuc_res, random_res = calculate_all_neighbours(morph_stats_cut, all_by_all_nucleus)
    # Cells where no cells meet the xyz criteria have values of -1
    with open(
            'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_all.pkl',
            'wb') as f:
        pickle.dump(nuc_res, f)
    random_res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_all_random.csv',
        index=False, sep='\t')

    # just cell stats
    morph_stats_cut = morph_stats.filter(regex='_cell')
    morph_stats_cut = morph_stats_cut.drop(columns=['label_id_cell'])
    nuc_res, random_res = calculate_all_neighbours(morph_stats_cut, all_by_all_nucleus)
    # Cells where no cells meet the xyz criteria have values of -1
    with open(
            'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_cell.pkl',
            'wb') as f:
        pickle.dump(nuc_res, f)
    random_res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_cell_random.csv',
        index=False, sep='\t')

    # just nuclei stats
    morph_stats_cut = morph_stats.filter(regex='_nucleus')
    morph_stats_cut = morph_stats_cut.drop(columns=['label_id_nucleus'])
    nuc_res, random_res = calculate_all_neighbours(morph_stats_cut, all_by_all_nucleus)
    # Cells where no cells meet the xyz criteria have values of -1
    with open(
            'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_nuc.pkl',
            'wb') as f:
        pickle.dump(nuc_res, f)
    random_res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_nuc_random.csv',
        index=False, sep='\t')

    # chromatin shape independent
    morph_stats_cut = morph_stats.filter(regex='_(het|eu)')
    morph_stats_cut_2 = morph_stats_cut.filter(regex='texture_')
    intensity_cols = np.array(['intensity_mean_het_nucleus', 'intensity_mean_eu_nucleus',
                               'intensity_st_dev_het_nucleus', 'intensity_st_dev_eu_nucleus',
                               'intensity_median_het_nucleus', 'intensity_median_eu_nucleus',
                               'intensity_iqr_het_nucleus', 'intensity_iqr_eu_nucleus'])
    cols_to_keep = np.append(intensity_cols, np.array(morph_stats_cut_2.columns))
    criteria = np.isin(np.array(morph_stats.columns), cols_to_keep)
    morph_stats_cut = morph_stats.loc[:, criteria]
    nuc_res, random_res = calculate_all_neighbours(morph_stats_cut, all_by_all_nucleus)
    # Cells where no cells meet the xyz criteria have values of -1
    with open(
            'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_chrom.pkl',
            'wb') as f:
        pickle.dump(nuc_res, f)
    random_res.to_csv(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_chrom_random.csv',
        index=False, sep='\t')


if __name__ == '__main__':
    main()
