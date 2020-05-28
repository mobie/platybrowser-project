import numpy as np
import pandas as pd


def add_trace_id_to_cell_table():
    cell_table_path = f'../../data/1.0.1/tables/sbem-6dpf-1-whole-segmented-cells/default.csv'
    trace_table_path = f'../../data/1.0.1/tables/sbem-6dpf-1-whole-traces/default.csv'

    cell_table = pd.read_csv(cell_table_path, sep='\t')
    trace_table = pd.read_csv(trace_table_path, sep='\t')

    trace_label_ids = trace_table['label_id'].values.astype('int')
    cell_ids = trace_table['cell_id'].values.astype('int')

    n_labels = len(cell_table)
    trace_ids = np.zeros(n_labels, dtype='uint32')

    trace_ids[cell_ids] = trace_label_ids

    table = cell_table.values
    table = np.concatenate([table, trace_ids[:, None]], axis=1)

    cols = list(cell_table.columns)
    cols.append('trace_id')
    print(cols)
    cell_table = pd.DataFrame(table, columns=cols)
    cell_table.to_csv(cell_table_path, index=False, sep='\t')


if __name__ == '__main__':
    add_trace_id_to_cell_table()
