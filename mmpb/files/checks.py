import os
import pandas as pd
import z5py

from pybdv.metadata import get_data_path


# TODO check more attributes in the xml to make sure that this actually is
# a bdv format file
def check_bdv(path):
    ext = os.path.splitext(path)[1]
    if ext != '.xml':
        return False
    h5_path = get_data_path(path, return_absolute_path=True)
    if not os.path.exists(h5_path):
        return False
    return True


def check_csv(path):
    try:
        table = pd.read_csv(path, sep='\t')
    except Exception:
        return False
    cols = table.columns
    if 'label_id' not in cols:
        return False
    return True


# TODO refactor to paintera_tools
def check_paintera(paintera_project):
    try:
        path, key = paintera_project
    except TypeError:
        return False
    try:
        f = z5py.File(path, 'r')
        group = f[key]
        # check for expected paintera keys
        for kk in ('data', 'label-to-block-mapping', 'unique-labels'):
            if kk not in group:
                return False
    except Exception:
        return False
    return True


# TODO check that we have all the mandatory colomns in
# the default table
def check_tables(table_dict):
    if 'default' not in table_dict:
        return False
    for path in table_dict.values():
        if not check_csv(path):
            return False
    return True
