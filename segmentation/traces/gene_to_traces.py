import os
import pandas as pd

ROOT = '../../data'


def gene_to_ids(version, gene_name):
    # get the normal table to check which ones are actually cells
    table = os.path.join(ROOT, version, 'tables',
                         'sbem-6dpf-1-whole-segmented-cells', 'vc_assignments.csv')
    table = pd.read_csv(able, sep='\t')

    # get the cells assigned to our gene
    gene_table = os.path.join(ROOT, version, 'tables',
                              'sbem-6dpf-1-whole-segmented-cells', 'vc_assignments.csv')
    gene_table = pd.read_csv(gene_table, sep='\t')

    return cell_ids


# we use the nuclei coordiantes
def ids_to_coordinates(version, cell_ids):
    cell_table = os.path.join(ROOT, version, 'tables',
                              'sbem-6dpf-1-whole-segmented-cells', 'vc_assignments.csv')


def gene_to_traces(version, gene_name, out):
    cell_ids = gene_to_ids(version, gene_name)
    center_coordinates = ids_to_coordinates(version, cell_ids)


if __name__ == '__main__':
    version = '1.0.1'
    name = 'Phc2'
    out = './traces_Phc2.nmx'
    gene_to_traces(version, name, out)
