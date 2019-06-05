#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python
# TODO new platy-browser env

import os
import numpy as np
import luigi
import z5py

from cluster_tools.node_labels import NodeLabelWorkflow
from .util import write_csv


def object_labels(seg_path, seg_key,
                  input_path, input_key, prefix,
                  tmp_folder, target, max_jobs):
    task = NodeLabelWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    out_path = os.path.join(tmp_folder, 'data.n5')
    out_key = 'node_labels_%s' % prefix

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             ws_path=seg_path, ws_key=seg_key,
             input_path=input_path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix=prefix, )
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Node labels for %s" % prefix)

    with z5py.File(out_path, 'r') as f:
        data = f[out_key][:]
    return data


def map_objects(label_ids, seg_path, seg_key, map_out,
                map_paths, map_keys, map_names,
                tmp_folder, target, max_jobs):
    assert len(map_paths) == len(map_keys) == len(map_names)

    data = []
    for map_path, map_key, prefix in zip(map_paths, map_keys, map_names):
        labels = object_labels(seg_path, seg_key,
                               map_path, map_key, prefix,
                               tmp_folder, target, max_jobs)
        data.append(labels[:, None])

    col_names = ['label_id'] + map_names
    data = np.concatenate([label_ids[:, None]] + data, axis=0)
    write_csv(map_out, data, col_names)
