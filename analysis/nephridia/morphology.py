import os
import numpy as np
import pandas as pd
from mmpb.analysis.nephridia import filter_by_size, match_cilia_to_cells

cell_ids = [24449, 22584, 21904, 21590, 21594, 21595, 21910, 21911, 21915]
print("Number of cells:", len(cell_ids))

tmp_path = './cilia_ids.npy'
if os.path.exists(tmp_path):
    cilia_ids = np.load(tmp_path)
else:
    cell_table = '../data/0.5.2/tables/sbem-6dpf-1-whole-segmented-cells/default.csv'
    cilia_path = '../data/0.5.2/segmentations/sbem-6dpf-1-whole-segmented-cilia.h5'
    cilia_key = 't00000/s00/2/cells'
    cilia_res = [0.1, 0.04, 0.04]
    cilia_ids = match_cilia_to_cells(cell_ids, cell_table,
                                     cilia_path, cilia_key, cilia_res)
    np.save(tmp_path, cilia_ids)

table_path = '../data/0.5.2/tables/sbem-6dpf-1-whole-segmented-cilia/default.csv'
table_path2 = '../data/0.5.2/tables/sbem-6dpf-1-whole-segmented-cilia/cilia.csv'

table1 = pd.read_csv(table_path, sep='\t')
table2 = pd.read_csv(table_path2, sep='\t')
table2 = table2[['length', 'diameter_mean']]
table = pd.concat([table1, table2], axis=1)
assert len(table1) == len(table2) == len(table)

table.set_index('label_id')
table = table[3:]
table = table.loc[table['length'] > 0.1]
ids = table['label_id'].values
ids = ids[np.isin(ids, cilia_ids)]
print(ids)
table = table.loc[ids]

# plot_sizes(table)
size_threshold = 5000
table = filter_by_size(table, size_threshold)
print("Number of cilia:", len(table))

lens = table["length"].values
for idd, leng in zip(ids, lens):
    print(idd, leng)
print("Average len:", np.mean(lens), "+-", np.std(lens))

diameters = table["diameter_mean"].values
print("Average diameter:", np.mean(diameters), "+-", np.std(diameters))
