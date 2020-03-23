#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import json
import os
import numpy as np
from mmpb.export import extract_neuron_traces_from_nmx, traces_to_volume, make_traces_table


def get_resolution(scale, use_nm=True):
    if use_nm:
        res0 = [25, 10, 10]
        res1 = [25, 20, 20]
    else:
        res0 = [0.025, 0.01, 0.01]
        res1 = [0.025, 0.02, 0.02]
    resolutions = [res0] + [[re * (2 ** (i)) for re in res1] for i in range(5)]
    return np.array(resolutions[scale])


def get_traces(folder):
    tmp_path = 'traces.json'
    if os.path.exists(tmp_path):
        with open(tmp_path) as f:
            traces = json.load(f)
        traces = {int(k): v for k, v in traces.items()}
        return traces

    traces = extract_neuron_traces_from_nmx(folder)
    with open(tmp_path, 'w') as f:
        json.dump(traces, f)
    return traces


def export_traces():
    folder = '/g/kreshuk/data/arendt/platyneris_v1/tracings/kevin'
    ref_path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    seg_out_path = '../../data/rawdata/sbem-6dpf-1-whole-traces.n5'
    table_out_path = '../../data/rawdata/tables/sbem-6dpf-1-whole-traces/default.csv'

    ref_scale = 3
    cell_seg_info = {'path': '../../data/1.0.0/images/local/sbem-6dpf-1-whole-segmented-cells.xml',
                     'scale': 2}
    nucleus_seg_info = {'path': '../../data/0.0.0/images/local/sbem-6dpf-1-whole-segmented-nuclei.xml',
                        'scale': 0}

    print("Extracting traces ...")
    traces = get_traces(folder)
    print("Found", len(traces), "traces")

    resolution = get_resolution(ref_scale)
    n_scales = 4
    scale_factors = n_scales * [[2, 2, 2]]
    print("Write trace volume ...")
    traces_to_volume(traces, ref_path, ref_scale, seg_out_path, resolution, scale_factors)

    print("Make table for traces ...")
    make_traces_table(traces, ref_scale, resolution, table_out_path,
                      {'cell': cell_seg_info, 'nucleus': nucleus_seg_info})


if __name__ == '__main__':
    export_traces()
