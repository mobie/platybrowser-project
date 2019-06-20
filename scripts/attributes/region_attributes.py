import os
import numpy as np
import h5py

from .util import write_csv, node_labels
from ..files import get_h5_path_from_xml


def write_region_table(label_ids, label_list, semantic_mapping_list, out_path):
    assert len(label_list) == len(semantic_mapping_list)
    n_labels = len(label_ids)
    assert all(len(labels) == n_labels for labels in label_list)

    col_names = ['label_id'] + [name for mapping in semantic_mapping_list
                                for name in mapping.keys()]
    n_cols = len(col_names)
    table = np.zeros((n_labels, n_cols))
    table[:, 0] = label_ids

    # print()
    # print()
    # print()
    col_offset = 1
    for labels, mapping in zip(label_list, semantic_mapping_list):
        for map_name, map_ids in mapping.items():
            with_label = np.in1d(labels, map_ids)
            # print(map_name)
            # print("Number of mapped:", with_label.sum())
            table[:, col_offset] = with_label
            col_offset += 1
    # print()
    # print()
    # print()

    write_csv(out_path, table, col_names)


def region_attributes(seg_path, region_out, segmentation_folder,
                      label_ids, tmp_folder, target, max_jobs):

    key0 = 't00000/s00/0/cells'

    # 1.) compute the mapping to carved regions
    #
    carved_path = os.path.join(segmentation_folder, 'em-segmented-tissue-labels.xml')
    carved_path = get_h5_path_from_xml(carved_path)
    carved_labels = node_labels(seg_path, key0,
                                carved_path, key0,
                                'carved-regions', tmp_folder,
                                target, max_jobs)
    # load the mapping of ids to semantics
    with h5py.File(carved_path) as f:
        names = f['semantic_names'][:]
        ids = f['semantic_mapping'][:]
    semantics_to_carved_ids = {name: idx.tolist()
                               for name, idx in zip(names, ids)}

    # 2.) compute the mapping to muscles
    muscle_path = os.path.join(segmentation_folder, 'em-segmented-muscles.xml')
    muscle_path = get_h5_path_from_xml(muscle_path)
    muscle_labels = node_labels(seg_path, key0,
                                muscle_path, key0,
                                'muscle', tmp_folder,
                                target, max_jobs)
    semantic_muscle = {'muscle': [255]}

    # 3.) merge the mappings and write new table
    write_region_table(label_ids, [carved_labels, muscle_labels],
                       [semantics_to_carved_ids, semantic_muscle],
                       region_out)
