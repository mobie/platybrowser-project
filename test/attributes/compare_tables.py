import numpy as np


def load_table(table_file):
    table = np.genfromtxt(table_file, delimiter='\t', skip_header=1,
                          dtype='float32')
    return table


# tab1 = load_table('./tmp_morpho/table_cells.csv')
# tab2 = load_table('../../data/0.0.0/tables/em-segmented-cells-labels/morphology.csv')
tab1 = load_table('./tmp_morpho/table_nuclei.csv')
tab2 = load_table('../../data/0.0.0/tables/em-segmented-nuclei-labels/morphology.csv')
print(tab1.shape)
print(tab2.shape)

close = np.isclose(tab1, tab2)
print(close.shape)
print(close.sum(axis=0))
