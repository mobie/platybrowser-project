import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_specificity_scores(col, ganglion_col):
    """ col is the boolean column with nrow == number of cells representing expression of gene
    or membership of cluster. Ganglion col is the boolean column representing membership of a ganglion.
    len(col) == len(ganglion_col) """

    no_cells_in_ganglia = np.sum(ganglion_col)
    no_expressing = np.sum(col)
    no_expressing_in_ganglia = np.sum(np.logical_and(col, ganglion_col))

    percent_of_expression_in_region = no_expressing_in_ganglia / no_expressing
    percent_of_region_covered = no_expressing_in_ganglia / no_cells_in_ganglia

    specificity_arithmetic_mean = (percent_of_expression_in_region + percent_of_region_covered) / 2
    specificity_geometric_mean = np.sqrt(percent_of_expression_in_region * percent_of_region_covered)
    specificity_multiplication = percent_of_expression_in_region * percent_of_region_covered
    specificity_f1 = 2 * ((percent_of_expression_in_region * percent_of_region_covered) / (
                percent_of_expression_in_region + percent_of_region_covered))

    return [no_expressing, no_expressing_in_ganglia, no_cells_in_ganglia, percent_of_expression_in_region,
            percent_of_region_covered, specificity_arithmetic_mean, specificity_geometric_mean,
            specificity_multiplication, specificity_f1]


def plot_specificity(plotting_frame, path, yaxis, top=None, fontsize=12):
    """ Can set top to filter for just top e.g. 10 results (top=10), otherwise will plot all results"""

    if top is not None:
        plotting_frame = plotting_frame.iloc[0:top, :]
        plt.figure(figsize=(top, 30))

    else:
        plt.figure(figsize=(30, 30))

    sns.set()
    sns.set_style('white')
    x = sns.barplot(x='id', y=yaxis, color="#3498db", data=plotting_frame)
    x.set_xticklabels(plotting_frame['id'], rotation=90)
    plt.rcParams.update({'font.size': fontsize})
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel('Genes and clusters')
    plt.ylabel(yaxis)
    plt.savefig(path)
    plt.close()


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.specificity_dir):
        os.makedirs(snakemake.output.specificity_dir)

    gene_table = pd.read_csv(snakemake.input.full_with_ids, sep='\t')
    ganglia = pd.read_csv(snakemake.input.ganglia, sep='\t')
    clustering = pd.read_csv(snakemake.input.clustering, sep='\t')

    if 'unique' in snakemake.params.gene_assign:
        # make clustering table mapped back to label id (from unique id) and in same order as full with ids
        cut = gene_table[['label_id', 'unique_id']]
        clustering_col_name = clustering.columns[-1]
        table = cut.join(clustering.set_index('unique_id'), on='unique_id', how='left')
        clusters = table[clustering_col_name]

    else:
        clusters = clustering.iloc[:, 1]

    # re-order ganglion table so order of rows == order of rows in gene table
    ganglion_cut = gene_table.loc[:, gene_table.columns == 'label_id']
    ganglion_cut = ganglion_cut.join(ganglia.set_index('label_id'), on='label_id', how='left')

    # replace any nans with 0
    ganglion_cut = ganglion_cut.fillna(0)

    # remove labels if present
    for var in ['label_id', 'unique_id']:
        if var in gene_table.columns:
            gene_table = gene_table.drop(columns=[var])

    ganglia_ids = np.unique(ganglion_cut['ganglion_id'])

    for i in ganglia_ids:
        print(i)
        result = []
        ganglia_col = ganglion_cut['ganglion_id'] == i

        # for each gene calculate specificity - 0.5 overlap cutoff
        for gene in gene_table:
            col = gene_table[gene] > 0.5
            result.append(calculate_specificity_scores(col, ganglia_col))

        # for each cluster calculate specificty
        for cluster_id in np.unique(clusters):
            col = clusters == cluster_id
            result.append(calculate_specificity_scores(col, ganglia_col))

        # put all results in dataframe
        result = pd.DataFrame.from_records(result)
        result.columns = ['no_expressing', 'no_expressing_in_ganglia', 'no_cells_in_ganglia',
                          'percent_of_expression_in_region',
                          'percent_of_region_covered', 'specificity_arithmetic_mean', 'specificity_geometric_mean',
                          'specificity_multiplication', 'specificity_f1']
        result['id'] = np.append(np.array(gene_table.columns), np.unique(clusters))
        result.to_csv(os.path.join(snakemake.output.specificity_dir, '%s_ganglia_specificity_scores.csv' % i),
                      index=False, sep='\t')

        # drop any rows containing nan values - this can happen when a gene is expressed in no cells under the
        # 0.5 threshold
        result = result.dropna()

        # plot all results
        for var in ['percent_of_expression_in_region',
                    'percent_of_region_covered', 'specificity_arithmetic_mean', 'specificity_geometric_mean',
                    'specificity_multiplication', 'specificity_f1']:
            plotting_frame = result[['id', var]]
            plotting_frame = plotting_frame.sort_values(by=[var], ascending=False)
            # only show cells with values > 0 to avoid cluttering up the graph
            plotting_frame = plotting_frame.loc[plotting_frame[var] > 0, :]

            folder_path = os.path.join(snakemake.output.specificity_dir, var)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            plot_path = os.path.join(folder_path, '%s.png' % i)
            plot_specificity(plotting_frame, plot_path, var)


if __name__ == '__main__':
    main()
