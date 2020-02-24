import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def save_tables(table):
    """ Saves filtered and normalised tables """
    table.to_csv(snakemake.output.filtered, index=False, sep='\t')
    norm_data = StandardScaler().fit_transform(table)
    np.savetxt(snakemake.output.normalise, norm_data, delimiter='\t')


# TODO - filtered tables only get used for the naming - perhaps just make the normalised tables named dataframes
# so this isn't so awkward here where the filtered table is different to the PCA
def save_tables_pca(table):
    """ Runs PCA - saves one table (filtered) with col names, and one without.
     Keep components that cumulatively describe at least 80% of variance """
    table_column_names = table.columns

    norm_data = StandardScaler().fit_transform(table)
    pca = PCA().fit(norm_data)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold_variance = 0.8
    max_index = np.where(explained_variance > threshold_variance)[0][0]

    components = pd.DataFrame(pca.components_)
    components.columns = table_column_names
    components.to_csv(snakemake.params.components, index=False, sep='\t')

    pca = PCA().fit_transform(norm_data)
    pca = pca[:, 0:max_index + 1]
    np.savetxt(snakemake.output.normalise, pca, delimiter='\t')

    pca = pd.DataFrame(pca)
    pca.columns = ['PC_%s' % i for i in range(0, pca.shape[1])]
    pca.to_csv(snakemake.output.filtered, index=False, sep='\t')


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    table = pd.read_csv(snakemake.input.table, sep='\t')

    # remove label ids
    table = table.drop(columns=['label_id_cell', 'label_id_nucleus'])

    subset = snakemake.params.sub

    if subset == 'all':
        save_tables(table)

    elif subset == 'pca_all':
        save_tables_pca(table)

    elif subset == 'shape_all':
        table = table.filter(regex='shape_*')
        save_tables(table)

    elif subset == 'cell_all':
        table = table.filter(regex='_cell')
        save_tables(table)

    elif subset == 'nuclei_all':
        table = table.filter(regex='_nucleus')
        save_tables(table)

    elif subset == 'chromatin_all':
        table = table.filter(regex='_(het|eu)')
        save_tables(table)

    elif subset == 'nuclei_shape':
        table = table.filter(regex='shape_*')
        table = table.filter(regex='_nucleus')
        save_tables(table)

    elif subset == 'nuclei_shape_no_chromatin':
        table = table.filter(regex='shape_*')
        table = table.filter(regex='_nucleus')
        table2 = table.filter(regex='_(het|eu)')
        for col in table2.columns:
            if col in table.columns:
                table = table.drop(columns=[col])

        save_tables(table)

    elif subset == 'shape_chromatin':
        table = table.filter(regex='shape_*')
        table = table.filter(regex='_(het|eu)')
        save_tables(table)

    elif subset == 'shape_independent_chromatin':
        table = table.filter(regex='_(het|eu)')
        table2 = table.filter(regex='texture_')
        intensity_cols = np.array(['intensity_mean_het_nucleus', 'intensity_mean_eu_nucleus',
                                   'intensity_st_dev_het_nucleus', 'intensity_st_dev_eu_nucleus',
                                   'intensity_median_het_nucleus', 'intensity_median_eu_nucleus',
                                   'intensity_iqr_het_nucleus', 'intensity_iqr_eu_nucleus'])
        cols_to_keep = np.append(intensity_cols, np.array(table2.columns))
        criteria = np.isin(np.array(table.columns), cols_to_keep)
        table = table.loc[:, criteria]
        save_tables(table)

    elif subset == 'shape_no_chromatin':
        table = table.filter(regex='shape_*')
        table2 = table.filter(regex='_(het|eu)')
        for col in table2.columns:
            if col in table.columns:
                table = table.drop(columns=[col])
        save_tables(table)


if __name__ == '__main__':
    main()
