import numpy as np
from .util import write_csv, node_labels


def map_objects(label_ids, seg_path, seg_key, map_out,
                map_paths, map_keys, map_names,
                tmp_folder, target, max_jobs):
    assert len(map_paths) == len(map_keys) == len(map_names)

    data = []
    for map_path, map_key, prefix in zip(map_paths, map_keys, map_names):
        labels = node_labels(seg_path, seg_key,
                             map_path, map_key, prefix,
                             tmp_folder, target, max_jobs)
        data.append(labels[:, None])

    col_names = ['label_id'] + map_names
    data = np.concatenate([label_ids[:, None]] + data, axis=1)
    write_csv(map_out, data, col_names)
