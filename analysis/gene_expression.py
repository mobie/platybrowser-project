#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
import os
import json
import numpy as np
from scripts import get_latest_version
from scripts.analysis import get_cells_expressing_genes


def count_gene_expression(gene_names, threshold, version, query):

    if args.query != '':
        assert os.path.exists(query)
        with open(query) as f:
            query_ids = np.array(json.load(f))
    else:
        query_ids = None

    # path are hard-coded, so we need to change the pwd to '..'
    os.chdir('..')
    try:
        if version == '':
            version = get_latest_version()

        # TODO enable using vc assignments once we have them on master
        table_path = 'data/%s/tables/sbem-6dpf-1-whole-segmented-cells-labels/genes.csv' % version
        ids = get_cells_expressing_genes(table_path, threshold, gene_names)

        if query_ids is not None:
            ids = ids[np.isin(ids, query_ids)]

        n = len(ids)
        print("Found", n, "cells expressing:", ",".join(gene_names))

    except Exception as e:
        os.chdir('analysis')
        raise e

    os.chdir('analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute number of cells co-expressing genes.')
    parser.add_argument('gene_names', type=str, nargs='+',
                        help='Names of the genes for which to express co-expression.')
    parser.add_argument('--threshold', type=float, default=.5,
                        help='Threshold to count gene expression. Default is 0.5.')
    parser.add_argument('--version', type=str, default='',
                        help='Version of the platy browser data. Default is latest.')
    parser.add_argument('--query', type=str, default='',
                        help='Path to json with list of cell ids to restrict the query to.')

    args = parser.parse_args()
    count_gene_expression(args.gene_names, args.threshold, args.version, args.query)
