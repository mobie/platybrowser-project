import os
import h5py

from .base_attributes import base_attributes
from ..files import get_h5_path_from_xml


def get_seg_path(folder, name, key):
    xml_path = os.path.join(folder, 'segmentations', '%s.xml' % name)
    path = get_h5_path_from_xml(xml_path, return_absolute_path=True)
    assert os.path.exists(path), path
    with h5py.File(path, 'r') as f:
        print(path)
        assert key in f, "%s not in %s" % (key, str(list(f.keys())))
    return path


def make_cell_tables(folder, name, tmp_folder, resolution,
                     target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_path = get_seg_path(folder, name)
    seg_key = 't00000/s00/0/cells'

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    base_attributes(seg_path, seg_key, base_out, resolution,
                    tmp_folder, target=target, max_jobs=max_jobs,
                    correct_anchors=True)

    # TODO
    # make table with mapping to other objects
    # nuclei
    # cellular models

    # TODO additional tables:
    # gene mapping
    # ???


def make_nucleus_tables(folder, name, tmp_folder, resolution,
                        target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = 't00000/s00/0/cells'
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    base_attributes(seg_path, seg_key, base_out, resolution,
                    tmp_folder, target=target, max_jobs=max_jobs,
                    correct_anchors=True)

    # TODO
    # make table with mapping to other objects
    # cells

    # TODO additional tables:
    # kimberly's nucleus attributes
