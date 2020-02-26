import os
import json

import numpy as np
import nifty.tools as nt
from elf.io import open_file


def add_max_id(path, key, max_id=None):
    with open_file(path) as f:
        ds = f[key]
        if max_id is None:
            data = ds[:]
            max_id = int(data.max())
        ds.attrs['maxId'] = max_id


def is_h5_file(path):
    return os.path.splitext(path)[1].lower() in ('.h5', '.hdf5', '.hdf')


def read_resolution(paintera_path, paintera_key, to_um=True):
    with open_file(paintera_path, 'r') as f:
        g = f[paintera_key]
        attrs = g['data'].attrs
        if 'resolution' not in attrs:
            raise ValueError("Invalid paintera container")
        resolution = attrs['resolution']
    # convert to z,y,x from java's xyz conventin
    resolution = resolution[::-1]
    # convert from nm to um
    if to_um:
        resolution = [float(res) / 1000. for res in resolution]
    return resolution


def propagate_lut(lut_path, ids):
    with open(lut_path) as f:
        lut = json.load(f)
    lut = {int(k): v for k, v in lut.items()}
    return nt.takeDict(lut, np.array(ids, dtype='uint32')).tolist()


def propagate_ids(root, src_version, trgt_version, seg_name, ids):
    """ Propagate list of ids from source version to target version.
    """
    version_file = os.path.join(root, 'versions.json')
    with open(version_file) as f:
        versions = json.load(f)
    versions.sort()

    lut_name = 'new_id_lut_%s.json' % seg_name
    src_lut_file = os.path.join(root, src_version, 'misc', lut_name)
    if not os.path.exists(src_lut_file):
        raise ValueError("Src lut %s does not exist." % src_lut_file)
    trgt_lut_file = os.path.join(root, trgt_version, 'misc', lut_name)
    if not os.path.exists(trgt_lut_file):
        raise ValueError("Target lut %s does not exist." % trgt_lut_file)

    def get_abs_lut(lut):
        return os.path.abspath(os.path.realpath(lut))

    # follow links from src-lut to target lut and pick up
    # all existing luts on the way.
    luts = []
    exclude_luts = [get_abs_lut(src_lut_file)]
    lut = src_lut_file
    version = src_version
    while True:

        abs_lut = get_abs_lut(lut)
        if abs_lut not in exclude_luts:
            luts.append(abs_lut)
            exclude_luts.append(abs_lut)

        version_index = versions.index(version)
        version = versions[version_index + 1]
        lut = os.path.join(root, version, 'misc', lut_name)
        if version == trgt_version:
            abs_lut = get_abs_lut(lut)
            if abs_lut not in luts:
                luts.append(abs_lut)
            break

    def load_lut(lut_path):
        with open(lut_path) as f:
            lut = json.load(f)
        lut = {int(k): v for k, v in lut.items()}
        return lut

    luts = [load_lut(lut) for lut in luts]

    # propagate ids through all luts
    propagated = np.array(ids, dtype='uint64')
    for lut in luts:
        propagated = nt.takeDict(lut, propagated)
    return propagated.tolist()


def write_additional_table_file(table_folder):
    # get all the file names in the table folder
    file_names = os.listdir(table_folder)
    file_names.sort()

    # make sure we have the default table
    default_name = 'default.csv'
    if default_name not in file_names:
        raise RuntimeError("Did not find the default table ('default.csv') in the table folder %s" % table_folder)

    # don't write anything if we don't have additional tables
    if len(file_names) == 1:
        return

    # write file for the additional tables
    out_file = os.path.join(table_folder, 'additional_tables.txt')
    with open(out_file, 'w') as f:
        for name in file_names:
            ext = os.path.splitext(name)[1]
            # only add csv files
            if ext != '.csv':
                continue
            # don't add the default table
            if name == 'default.csv':
                continue
            f.write(name + '\n')
