import json
import numpy as np
import pandas as pd
import nifty.tools as nt

# the current table is w.r.t.
# cilia ids from 0.5.3 and we need 0.6.2
# cell ids from 0.3.1 and we need 0.6.1


def propagate_table():
    table_path = './20191030_table_ciliaID_cellID'
    table = pd.read_csv(table_path, sep='\t')
    cilia_ids = table['cilia_id'].values.astype('uint32')
    cell_ids = table['cell_id'].values
    cell_ids[np.isinf(cell_ids)] = 0
    cell_ids[np.isnan(cell_ids)] = 0
    cell_ids = cell_ids.astype('uint32')

    cell_id_mapping = {24024: 24723, 22925: 23531, 22700: 23296, 22699: 23295,
                       22584: 23199, 22515: 23132, 22182: 22827, 22181: 22826,
                       21915: 22549, 21911: 22546, 21910: 22545, 21904: 22541,
                       21594: 22214, 21590: 22211, 0: 0}
    unique_vals, unique_counts = np.unique(list(cell_id_mapping.values()), return_counts=True)
    print(unique_vals)
    assert (unique_counts == 1).all()
    cell_ids = nt.takeDict(cell_id_mapping, cell_ids)

    cilia_id_mapping = '../../data/0.6.2/misc/new_id_lut_sbem-6dpf-1-whole-segmented-cilia-labels.json'
    with open(cilia_id_mapping) as f:
        cilia_id_mapping = json.load(f)
    cilia_id_mapping = {int(k): v for k, v in cilia_id_mapping.items()}

    cilia_ids = [cilia_id_mapping.get(cil_id, 0) for cil_id in cilia_ids]
    cilia_ids = np.array(cilia_ids)

    valid_mask = ~(cilia_ids == 0)
    cilia_ids = cilia_ids[valid_mask]
    cell_ids = cell_ids[valid_mask]
    sorter = np.argsort(cilia_ids)
    cilia_ids = cilia_ids[sorter]
    cell_ids = cell_ids[sorter]

    table_out = './20191030_table_ciliaID_cellID_out'
    new_table = np.concatenate([cilia_ids[:, None], cell_ids[:, None]], axis=1)
    new_table = pd.DataFrame(new_table, columns=['label_id', 'cell_id'])
    new_table.to_csv(table_out, sep='\t', index=False)


if __name__ == '__main__':
    propagate_table()
