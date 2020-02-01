import pandas as pd

def_path = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv.bkp'
t1 = pd.read_csv(def_path, sep='\t')
t2 = pd.read_csv('../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/cilia.csv', sep='\t')

l1 = t1[['label_id']].values
l2 = t2[['label_id']].values
assert (l1 == l2).all()

t = pd.concat([t1, t2['cell_id']], axis=1)
def_path = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
t.to_csv(def_path, index=False, sep='\t')
