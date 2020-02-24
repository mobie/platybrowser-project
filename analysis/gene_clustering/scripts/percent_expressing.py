import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    gene = pd.read_csv(snakemake.input.gene, sep='\t')

    # remove label ids if present
    for var in ['label_id', 'unique_id']:
        if var in gene.columns:
            gene = gene.drop(columns=[var])

    no_exp = gene.sum(axis=0)
    plotting_frame = pd.DataFrame({'gene': gene.columns, 'no_exp': no_exp})
    plotting_frame = plotting_frame.sort_values(by=['no_exp'], ascending=False)

    plt.figure(figsize=(30, 30))
    sns.set()
    sns.set_style('white')
    x = sns.barplot(x='gene', y='no_exp', color="#3498db", data=plotting_frame)
    x.set_xticklabels(plotting_frame['gene'], rotation=90, fontsize=12)
    plt.yticks(np.arange(0, max(no_exp) + 1, 50))
    plt.xlabel('Genes')
    plt.ylabel('Number of cells expressing - after binarisation')
    plt.savefig(snakemake.output.expressing)


if __name__ == '__main__':
    main()
