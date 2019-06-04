import os


def make_cell_tables(folder, name):
    seg_file = os.path.join('folder', 'segmentations', '%s.h5' % name)
    table_folder = os.path.join(folder, 'tables', name)
    os.mkdir(table_folder)


def make_nucleus_tables(folder, name):
    seg_file = os.path.join('folder', 'segmentations', '%s.h5' % name)
    table_folder = os.path.join(folder, 'tables', name)
    os.mkdir(table_folder)
