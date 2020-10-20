import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_y_values(result_array):
    """ result array is a numpy array """
    total_no = len(result_array)

    # drop any -1 which indicate cells where no partner was found that matched the xyz criteria
    result_array = result_array[result_array != -1]

    # step size / range for x axis
    x_axis = list(range(1, 10400, 1))
    y_axis = []

    for val in x_axis:
        # get fraction of total array that are below the current value
        cumulative = np.sum(result_array <= val) / total_no
        y_axis.append(cumulative)

    return y_axis


def calculate_y_values_random(result_df):
    results_random = []

    # for each row calculate corresponding y values
    for i in range(0, result_df.shape[0]):
        row = np.array(result_df.iloc[i, :])
        y_vals = calculate_y_values(row)
        results_random.append(y_vals)

    # take mean for each y value
    results_random = pd.DataFrame.from_records(results_random)
    mean_y = results_random.mean()  # over columns

    return mean_y


# all stats
with open(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_all.pkl',
        'rb') as f:
    all_result = pickle.load(f)
all_random = pd.read_csv(
    'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_all_random.csv',
    sep='\t')

x_axis = list(range(1, 10400, 1))
y_vals_all = calculate_y_values(np.array(all_result))
y_vals_all_random = calculate_y_values_random(all_random)
plt.scatter(x_axis, y_vals_all, label='all')
plt.scatter(x_axis, y_vals_all_random, label='all_random')
res_all = pd.DataFrame({'x': x_axis, 'y_all': y_vals_all, 'y_random': y_vals_all_random})

# cell stats
with open(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_cell.pkl',
        'rb') as f:
    cell_result = pickle.load(f)
cell_random = pd.read_csv(
    'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_cell_random.csv',
    sep='\t')

y_vals_cell = calculate_y_values(np.array(cell_result))
y_vals_cell_random = calculate_y_values_random(cell_random)
plt.scatter(x_axis, y_vals_cell, label='cell')
plt.scatter(x_axis, y_vals_cell_random, label='cell_random')
res_all['y_cell'] = y_vals_cell
res_all['y_cell_random'] = y_vals_cell_random

# nucleus stats
with open(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_nuc.pkl',
        'rb') as f:
    nuc_result = pickle.load(f)
nuc_random = pd.read_csv(
    'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_nuc_random.csv',
    sep='\t')

y_vals_nuc = calculate_y_values(np.array(nuc_result))
y_vals_nuc_random = calculate_y_values_random(nuc_random)
plt.scatter(x_axis, y_vals_nuc, label='nuc')
plt.scatter(x_axis, y_vals_nuc_random, label='nuc_random')
res_all['y_nuc'] = y_vals_nuc
res_all['y_nuc_random'] = y_vals_nuc_random

# chromatin stats
with open(
        'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_chrom.pkl',
        'rb') as f:
    chrom_result = pickle.load(f)
chrom_random = pd.read_csv(
    'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\1_0_1\\all_by_all_chrom_random.csv',
    sep='\t')

y_vals_chrom = calculate_y_values(np.array(chrom_result))
y_vals_chrom_random = calculate_y_values_random(chrom_random)
plt.scatter(x_axis, y_vals_chrom, label='chrom')
plt.scatter(x_axis, y_vals_chrom_random, label='chrom_random')
res_all['y_chrom'] = y_vals_chrom
res_all['y_chrom_random'] = y_vals_chrom_random

plt.xlabel('First neighbour that meets xyz criteria')
plt.ylabel('% of all cells')
plt.legend()
# plt.close()

# 200 neighbour maximum
plt.xlim(1, 200)

res_all.to_csv(
    'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\plot_values_1_0_1.csv',
    index=False, sep='\t')
