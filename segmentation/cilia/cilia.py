import json
import numpy as np
import pandas as pd


def map_cell_ids():
    new_cil_ids = '../data/0.5.3/misc/new_id_lut_sbem-6dpf-1-whole-segmented-cilia-labels.json'
    with open(new_cil_ids) as f:
        new_cil_ids = json.load(f)
    new_cil_ids = {int(k): v for k, v in new_cil_ids.items()}

    old_cell_mapping = '../data/0.5.2/misc/cilia_id_mapping.csv'
    old_cell_mapping = pd.read_csv(old_cell_mapping, sep='\t')
    names = old_cell_mapping.columns.values
    old_cell_mapping = old_cell_mapping.values
    old_cell_mapping = dict(zip(old_cell_mapping[:, 0], old_cell_mapping[:, 1]))

    n_new_cilia = max(new_cil_ids.keys()) + 1
    new_cell_mapping = np.zeros((n_new_cilia, 2), dtype='uint32')

    for new_cil_id, old_cil_id in new_cil_ids.items():
        new_cell_mapping[new_cil_id, 0] = new_cil_id
        cell_id = old_cell_mapping.get(old_cil_id, 0)
        new_cell_mapping[new_cil_id, 1] = cell_id
        if cell_id != 0:
            print(new_cil_id, old_cil_id, cell_id)

    new_cell_mapping = pd.DataFrame(new_cell_mapping, columns=names)
    out = '../data/0.5.3/tables/sbem-6dpf-1-whole-segmented-cilia-labels/cell_id_mapping.csv'
    new_cell_mapping.to_csv(out, sep='\t', index=False)


if __name__ == '__main__':
    map_cell_ids()
