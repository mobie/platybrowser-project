#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import numpy as np
import h5py
from scripts.attributes.base_attributes import base_attributes


def add_max_id():
    input_path = '../data/0.5.0/images/cellular-models-labels_180919_500nm.h5'
    input_key = 't00000/s00/0/cells'
    with h5py.File(input_path) as f:
        ds = f[input_key]
        data = ds[:]
        max_id = int(data.max())
        print("Found max id:", max_id)
        ds.attrs['maxId'] = max_id


def compute_vc_table():
    input_path = '../data/0.5.0/images/cellular-models-labels_180919_500nm.h5'
    input_key = 't00000/s00/0/cells'
    output_path = './vc_default.csv'
    tmp_folder = 'tmp_vc_table'
    target = 'local'
    max_jobs = 32

    resolution = [.5, .5, .5]
    base_attributes(input_path, input_key, output_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=False)


def check_ids():
    input_path = '../data/0.5.0/images/cellular-models-labels_180919_500nm.h5'
    input_key = 't00000/s00/0/cells'
    with h5py.File(input_path) as f:
        data = f[input_key][:]
    print(data.max())
    print(np.unique(data))


if __name__ == '__main__':
    add_max_id()
    compute_vc_table()
    # check_ids()
