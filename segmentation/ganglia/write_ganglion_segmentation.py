import os
import z5py
import numpy as np
import pandas as pd
import nifty.tools as nt
from pybdv import make_bdv
from mmpb.attributes.base_attributes import base_attributes


def write_ganglion_segmentation():
    seg_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                            '0.6.5/images/local/sbem-6dpf-1-whole-segmented-cells.n5')
    tab_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                            '0.6.6/tables/sbem-6dpf-1-whole-segmented-cells/ganglia_ids.csv')
    tab = pd.read_csv(tab_path, sep='\t')

    seg_key = 'setup0/timepoint0/s0'
    with z5py.File(seg_path) as f:
        ds = f[seg_key]
        max_id = int(ds.attrs['maxId']) + 1

    label_ids = tab['label_id'].values.astype('uint32')
    ganglion_labels = tab['ganglion_id'].values.astype('uint32')
    label_mapping = np.zeros(max_id, 'uint32')
    label_mapping[label_ids] = ganglion_labels

    seg_key = 'setup0/timepoint0/s3'
    out_path = './sbem-6dpf-1-whole-segmented-ganglia.n5'

    print("Reading segmentation ...")
    with z5py.File(seg_path, 'r') as f:
        ds = f[seg_key]
        ds.n_threads = 16
        seg = ds[:].astype('uint32')

    print("To ganglion segmentation ...")
    seg = nt.take(label_mapping, seg)
    seg = seg.astype('int16')
    print(seg.shape)

    print("Writing segmentation ...")
    n_scales = 4
    res = [.2, .16, .16]
    chunks = (128,) * 3
    downscale_factors = n_scales * [[2, 2, 2]]
    make_bdv(seg, out_path, downscale_factors,
             resolution=res, unit='micrometer',
             chunks=chunks, n_threads=16)

    with z5py.File(out_path) as f:
        ds = f['setup0/timepoint0/s0']
        ds.attrs['maxId'] = max_id


def make_default_table():
    input_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                              'rawdata/sbem-6dpf-1-whole-segmented-ganglia.n5')
    input_key = 'setup0/timepoint0/s0'
    output_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                               'rawdata/tables/sbem-6dpf-1-whole-segmented-ganglia/default.csv')
    resolution = [.2, .16, .16]
    tmp_folder = './tmp_ganglia'
    target = 'local'
    max_jobs = 48
    base_attributes(input_path, input_key, output_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=False)


if __name__ == '__main__':
    write_ganglion_segmentation()
    make_default_table()
