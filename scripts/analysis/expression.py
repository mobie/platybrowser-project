import numpy as np
import pandas as pd


def get_cells_expressing_genes(table_path, expression_threshold, gene_names):
    if isinstance(gene_names, str):
        gene_names = [gene_names]
    if not isinstance(gene_names, list):
        raise ValueError("Gene names must be a str or a list of strings")

    table = pd.read_csv(table_path, sep='\t')
    label_ids = table['label_id']

    columns = table.columns
    unmatched = set(gene_names) - set(columns)
    if len(unmatched) > 0:
        raise RuntimeError("Could not find gene names %s in table %s" % (", ".join(unmatched),
                                                                         table_path))

    # find logical and of columns expressing the genes
    expressing = np.logical_and.reduce(tuple(table[name] > expression_threshold
                                             for name in gene_names))
    # get ids of columns expressing all genes
    label_ids = label_ids[expressing].values
    return label_ids
