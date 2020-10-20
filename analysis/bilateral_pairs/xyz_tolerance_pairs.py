import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_distances(stats, sym_pairs_cut):
    """ Get absolute distances between absolute, y & z of stats table for each symmetric pair """
    result = []
    for pair_id in np.unique(sym_pairs_cut.pair_index):
        label_ids = np.array(sym_pairs_cut.label_id[sym_pairs_cut.pair_index == pair_id])
        if len(label_ids) == 2:
            res_line = []
            for stat in ['absolute', 'new_y', 'new_z']:
                res_line.append(np.abs(float(stats.loc[stats.label_id == label_ids[0], stat]) - float(
                    stats.loc[stats.label_id == label_ids[1], stat])))

            result.append(res_line)

    full_result = pd.DataFrame.from_records(result)
    full_result.columns = ['absolute', 'new_y', 'new_z']

    return full_result


def make_summary_df(distances_table):
    """ make dataframe summarising mean, standard deviation and mean +- 2sd for each stat """
    # means
    ms = distances_table.mean()

    # st devs
    sds = distances_table.std()

    mean_with_sds = []

    # count it in mean +- st dev
    for i in range(0, len(ms)):
        mean_with_sds.append(ms[i] + 2 * sds[i])

    summary = pd.DataFrame(
        {'stat': distances_table.columns, 'mean': ms, 'standard_deviation': sds, 'mean_with_two_sds': mean_with_sds})

    return summary


nuc_cell_mapping = pd.read_csv(
    'W:\\EM_6dpf_segmentation\\platy-browser-data\\data\\1.0.1\\tables\\sbem-6dpf-1-whole-segmented-cells\\cells_to_nuclei.csv',
    sep='\t')
nuc_cell_mapping = nuc_cell_mapping.loc[nuc_cell_mapping['nucleus_id'] != 0, :]

# cell label ids
sym_pairs = pd.read_csv(
    'W:\\EM_6dpf_segmentation\\platy-browser-data\\data\\1.0.1\\tables\\sbem-6dpf-1-whole-segmented-cells\\symmetric_cells.csv',
    sep='\t')
# remove rows for cells not in pairs
sym_pairs = sym_pairs.loc[sym_pairs.pair_index != 0.0, :]
print(sym_pairs.shape)

# remove pairs that don't have assigned nuclei
sym_pairs_cut = sym_pairs.loc[np.isin(sym_pairs['label_id'], nuc_cell_mapping['label_id']), :]
print(sym_pairs_cut.shape)

# average nuclei distance discrepancy
transformed_xyz_nuc = pd.read_csv(
    'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/prospr_space_nuclei_points_1_0_1.csv',
    sep='\t')
transformed_xyz_nuc = transformed_xyz_nuc.rename(columns={'label_id': 'nucleus_id'})
midline_dist_nuc = pd.read_csv(
    'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/distance_from_midline_nuclei_1_0_1.csv',
    sep='\t')
midline_dist_nuc = midline_dist_nuc.rename(columns={'label_id': 'nucleus_id'})

# remove zero label if present & make merged table with label id of cell
transformed_xyz_nuc = transformed_xyz_nuc.loc[transformed_xyz_nuc.nucleus_id != 0, :]
midline_dist_nuc = midline_dist_nuc.loc[midline_dist_nuc.nucleus_id != 0, :]
transformed_xyz_nuc = transformed_xyz_nuc.join(nuc_cell_mapping.set_index('nucleus_id'), on='nucleus_id', how='inner')
midline_dist_nuc = midline_dist_nuc.join(nuc_cell_mapping.set_index('nucleus_id'), on='nucleus_id', how='inner')
# join stats
stats_nuc = transformed_xyz_nuc[['label_id', 'new_x', 'new_y', 'new_z']]
stats_nuc = stats_nuc.join(midline_dist_nuc.set_index('label_id'), on='label_id', how='inner')
stats_nuc = stats_nuc[['label_id', 'new_x', 'new_y', 'new_z', 'absolute']]

# average cell distance discrepancy
transformed_xyz_cells = pd.read_csv(
    'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/prospr_space_cells_points_1_0_1.csv',
    sep='\t')
midline_dist_cells = pd.read_csv(
    'Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/distance_from_midline_cells_1_0_1.csv',
    sep='\t')
transformed_xyz_cells = transformed_xyz_cells.loc[transformed_xyz_cells.label_id != 0, :]
midline_dist_cells = midline_dist_cells.loc[midline_dist_cells.label_id != 0, :]
stats_cell = transformed_xyz_cells[['label_id', 'new_x', 'new_y', 'new_z']]
stats_cell = stats_cell.join(midline_dist_cells.set_index('label_id'), on='label_id', how='inner')
stats_cell = stats_cell[['label_id', 'new_x', 'new_y', 'new_z', 'absolute']]

distances_nuclei = get_distances(stats_nuc, sym_pairs_cut)
distances_cells = get_distances(stats_cell, sym_pairs_cut)

summary_nuclei = make_summary_df(distances_nuclei)
summary_cells = make_summary_df(distances_cells)

summary_nuclei.to_csv('Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/bilateral_distance_summary_nuclei_1_0_1.csv', index=False, sep='\t')
summary_cells.to_csv('Z:/Kimberly/Projects/SBEM_analysis/src/sbem_analysis/paper_code/files_for_midline_xyz/bilateral_distance_summary_cells_1_0_1.csv', index=False, sep='\t')
