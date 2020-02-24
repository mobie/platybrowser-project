import pandas as pd
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pack.Clustering import plot_clusters_on_umap
from pack.table_utils import make_binary_columns
from sklearn.preprocessing import StandardScaler

log_path = snakemake.log[0]
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_texture_fails(table):
    """ Filter out rows that fail any of the haralick texture statistics. This often happens with the fully dark nuclei
    as euchromatin is only segmented as an erroneous rim around the outside. The small number of pixels and odd shape
    of this rim means haralick feature computations fail """

    logger.info('%s rows before texture filtering' % table.shape[0])

    texture = table.filter(regex='texture')
    criteria = (texture == 0).any(axis=1)
    table = table.loc[np.invert(criteria), :]
    logger.info('%s rows after texture filtering' % table.shape[0])

    return table


def main():
    umap = np.loadtxt(snakemake.input.umap, delimiter='\t')

    chosen_k = int(snakemake.params.chosen_k)
    chosen_res = float(snakemake.params.chosen_res)

    if chosen_k == 0:
        raise ValueError('No chosen number of neighbours - k - for louvain')

    elif chosen_res == 0:
        raise ValueError('No chosen resolution for louvain')

    with open(snakemake.input.cluster_file, 'rb') as f:
        clustering = pickle.load(f)

    plot, no_clusters, no_singletons = plot_clusters_on_umap(umap, clustering, label=True)
    plot.set_title(
        'k_%s_res_%s__no_clusters__%s__no_singletons__%s' % (chosen_k, chosen_res, no_clusters, no_singletons),
        fontsize=20)
    plt.savefig(snakemake.output.fig)

    table = pd.read_csv(snakemake.input.filtered, sep='\t')
    col_name = 'chosen_k_%s_res_%s' % (chosen_k, chosen_res)
    table[col_name] = clustering

    # just keep label id and clusters
    if 'label_id' in table.columns:
        table = table[['label_id', col_name]]
    elif 'unique_id' in table.columns:
        table = table[['unique_id', col_name]]
    table.to_csv(snakemake.output.merged_table, index=False, sep='\t')

    # make morph and viz tables
    viz_table = pd.read_csv(snakemake.input.viz_table, sep='\t')
    full_with_ids = pd.read_csv(snakemake.input.full_with_ids, sep='\t')
    merged_morph = pd.read_csv(snakemake.input.merged_morph, sep='\t')
    merged_morph = filter_texture_fails(merged_morph)

    if 'unique' not in snakemake.params.gene_assign:
        viz_table['clusters'] = clustering

        # make table with label id & clusters for all cells
        full_with_ids['clusters'] = clustering
        table = full_with_ids[['label_id', 'clusters']]

    else:
        # make table with label id & clusters for all cells
        cut = full_with_ids[['label_id', 'unique_id']]
        table = cut.join(table.set_index('unique_id'), on='unique_id', how='left')
        table = table[['label_id', col_name]]
        table.columns = ['label_id', 'clusters']

        viz_table = viz_table.join(table.set_index('label_id'), on='label_id', how='left')

    make_binary_columns(viz_table, 'clusters', snakemake.output.viz_table)
    morph_table = table.join(merged_morph.set_index('label_id'), on='label_id', how='inner')
    morph_table.to_csv(snakemake.output.morph_table, index=False, sep='\t')

    # and a normalised version
    just_morph = morph_table.drop(columns=['label_id', 'clusters'])
    col_names = just_morph.columns.tolist()
    just_morph = StandardScaler().fit_transform(just_morph)
    just_morph = pd.DataFrame(data=just_morph)
    just_morph.columns = col_names

    # can have issues with index matches producing nan values - reset indices here
    just_morph.reset_index(drop=True, inplace=True)
    morph_table.reset_index(drop=True, inplace=True)

    just_morph.insert(0, 'clusters', morph_table['clusters'])
    just_morph.insert(0, 'label_id', morph_table['label_id'])
    just_morph.to_csv(snakemake.output.morph_table_normalised, index=False, sep='\t')


if __name__ == '__main__':
    main()
