import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pack.Clustering import plot_categorical_feature_on_umap, plot_continuous_feature_on_umap


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.region_dir):
        os.makedirs(snakemake.output.region_dir)

    umap = np.loadtxt(snakemake.input.umap, delimiter='\t')
    table = pd.read_csv(snakemake.input.table, sep='\t')
    region = pd.read_csv(snakemake.input.region, sep='\t')
    label = snakemake.params.label

    if 'unique' not in snakemake.params.gene_assign:
        table = table.loc[:, table.columns == 'label_id']
        table = table.join(region.set_index('label_id'), on='label_id', how='left')
        table = table.drop(columns=['label_id'])

    # assign unique id to its mode region - if multiple modes (i.e. multiple regions have same max value of cells
    # assigned) then just use whichever mode happens to be listed first (random)
    else:
        full_with_ids = pd.read_csv(snakemake.input.full_with_ids, sep='\t')
        unique_ids = table['unique_id']
        label_id_to_unique_id = full_with_ids[['label_id', 'unique_id']]

        table = []
        for id in unique_ids:
            # label id to unique id mapping for one unique id
            cut_label_id_to_unique_id = label_id_to_unique_id.loc[label_id_to_unique_id.unique_id == id, :]
            # find region rows that match that unique id
            criteria = np.isin(region.label_id, cut_label_id_to_unique_id.label_id)
            # if no matches i.e. none of those label ids correspond to a given region
            if np.sum(criteria) == 0:
                table.append(np.repeat(np.nan, region.shape[1] - 1))
            # otherwise assign region common to most label ids in that unique id
            else:
                cut_region = region.loc[criteria, :]
                cut_region = cut_region.drop(columns=['label_id'])

                most_often = cut_region.mode()
                # if multiple modes (i.e more than one row), just take whichever is listed first
                most_often = most_often.iloc[0, :]
                table.append(most_often)

        table = pd.DataFrame.from_records(table)
        col_names = region.columns
        col_names = col_names[col_names != 'label_id']
        table.columns = col_names

    # replace any nans with zero
    table = table.fillna(0)

    for val in list(table.columns):
        plot = plot_categorical_feature_on_umap(umap, table[val], label)
        if plot is None:
            logger.info('%s failed due to only having one category' % val)
            # can then pass to the continuous plotting, as this won't fail on only one category
            plot = plot_continuous_feature_on_umap(umap, table[val])
        plot.set_title(val, fontsize=20)
        path = os.path.join(snakemake.output.region_dir, val + '.png')
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    main()
