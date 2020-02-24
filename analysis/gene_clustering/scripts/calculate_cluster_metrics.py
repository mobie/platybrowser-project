import pandas as pd
import logging
import os
from sklearn.metrics import calinski_harabasz_score, silhouette_score, silhouette_samples
from pack.Clustering import silhouette_plot
import matplotlib.pyplot as plt


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    clusters_summary = pd.read_csv(snakemake.input.summary_table, sep='\t')
    raw_data = pd.read_csv(snakemake.input.raw_data, sep='\t')

    # remove label ids if present
    for var in ['label_id', 'unique_id']:
        if var in raw_data.columns:
            raw_data = raw_data.drop(columns=[var])

    if not os.path.isdir(snakemake.output.plots):
        os.makedirs(snakemake.output.plots)

    result = []
    for row in clusters_summary.itertuples(index=False):
        ch_score = calinski_harabasz_score(raw_data, row[3:])

        sh_score = silhouette_score(raw_data, row[3:])

        result.append([row[0], row[1], row[2], ch_score, sh_score])

        silhouette_plot(raw_data, row[3:],
                        os.path.join(snakemake.output.plots, '%s_%s.png' % (row[0], row[1])))

    result_tab = pd.DataFrame.from_records(result)
    result_tab.columns = ['k', 'res', 'no_clusters', 'calinski_harabasz', 'silhouette']

    plt.figure(figsize=(20, 20))
    plt.scatter(result_tab['no_clusters'], result_tab['calinski_harabasz'])
    plt.savefig(os.path.join(snakemake.output.plots, 'calinski_harabasz.png'))
    plt.close()

    result_tab.to_csv(snakemake.output.metrics_table, index=False, sep='\t')


if __name__ == '__main__':
    main()
