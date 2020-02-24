import pandas as pd
import logging
import umap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    table = pd.read_csv(snakemake.input.table, delimiter='\t')

    # drop label id
    for var in ['label_id', 'unique_id']:
        if var in table.columns:
            table = table.drop(columns=[var])

    k = int(snakemake.params.k)
    metric = snakemake.params.metric

    uma = umap.UMAP(n_neighbors=k, min_dist=0.1, n_components=2, metric=metric,
                    random_state=15).fit_transform(table)
    plt.figure(figsize=(30, 20))
    plt.scatter(uma[:, 0], uma[:, 1])
    plt.title('UMAP with default min_dist & n_neighbors %s' % k)
    plt.savefig(snakemake.output.fig)


if __name__ == '__main__':
    main()
