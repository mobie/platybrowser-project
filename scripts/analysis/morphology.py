import numpy as np
import pandas as pd


def get_region_ids(region_table_path, nucleus_table_path, region_name):
    region_table = pd.read_csv(region_table_path, sep='\t')
    region_mask = region_table[region_name].values == 1

    nucleus_table = pd.read_csv(nucleus_table_path, sep='\t')
    nucleus_mask = nucleus_table["nucleus_id"].values > 0

    label_mask = np.logical_and(region_mask, nucleus_mask)
    return np.where(label_mask)[0]


def get_morphology_attribute(table_path, attribute_name, query_ids=None):

    table = pd.read_csv(table_path, sep='\t')
    label_ids = table['label_id']

    if query_ids is None:
        return table[attribute_name].values
    else:
        label_mask = np.isin(label_ids.values, query_ids)
        return table[attribute_name][label_mask].values
