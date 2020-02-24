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

    chosen_k = int(snakemake.params.chosen_k)
    mindist = float(snakemake.params.mindist)
    metric = snakemake.params.metric

    if chosen_k == 0:
        raise ValueError('No chosen number of neighbours - k - for umap')

    uma = umap.UMAP(n_neighbors=chosen_k, min_dist=mindist, n_components=2, metric=metric,
                    random_state=15).fit_transform(table)
    plt.figure(figsize=(30, 20))
    plt.scatter(uma[:, 0], uma[:, 1])
    plt.title('UMAP with min_dist %s & n_neighbors %s' % (mindist, chosen_k))
    plt.savefig(snakemake.output.fig)


if __name__ == '__main__':
    main()
