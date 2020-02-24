import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import umap


def calc_umap(morph, gene_name, gene_column, dir):
    """
    Calculate UMAP for cells expressing gene (i.e. 1s in gene column) based on stats in the morph dataframe

    Args:
        morph [pd.DataFrame] - dataframe of morphology statistics
        gene_name [str] - name of gene
        gene_column [listlike] - binarised gene expression (0/1) with number of rows == number of rows in morph dataframe
        dir [str] - directory to save result to
    """
    if sum(gene_column) == 0:
        return

    # only keep rows that express the gene (== 1)
    morph_cut = morph.loc[np.array(gene_column), :]
    uma = umap.UMAP(n_components=2, metric='euclidean',
                    random_state=15).fit_transform(morph_cut)

    plt.figure(figsize=(30, 20))
    plt.scatter(uma[:, 0], uma[:, 1])
    plt.title('UMAP based on morph for gene %s' % gene_name)
    path = os.path.join(dir, gene_name + '.png')
    plt.savefig(path)
    plt.close()


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.out_dir):
        os.makedirs(snakemake.output.out_dir)

    gene = pd.read_csv(snakemake.input.genes, sep='\t')
    morph = pd.read_csv(snakemake.input.morph, sep='\t')
    gene = gene.drop(columns=['label_id_cell', 'label_id_nucleus'])

    gene_list = [gene[var] for var in gene.columns]

    with ProcessPoolExecutor() as e:
        results = list(
            e.map(calc_umap, [morph] * gene.shape[1], gene.columns, gene_list,
                  [snakemake.output.out_dir] * gene.shape[1]))


if __name__ == '__main__':
    main()
