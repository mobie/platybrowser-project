import os
import glob
import numpy as np
import pandas as pd
from elf.io import open_file
from pybdv.metadata import get_data_path

from .util import write_csv, node_labels, normalize_overlap_dict, get_seg_key_xml


def write_region_table(label_ids, label_list, semantic_mapping_list, out_path):
    assert len(label_list) == len(semantic_mapping_list)
    n_labels = len(label_ids)
    assert all(len(labels) == n_labels for labels in label_list)
    col_names = ['label_id'] + [name for mapping in semantic_mapping_list
                                for name in mapping.keys()]
    n_cols = len(col_names)
    table = np.zeros((n_labels, n_cols))
    table[:, 0] = label_ids

    col_offset = 1
    for labels, mapping in zip(label_list, semantic_mapping_list):
        for map_name, map_ids in mapping.items():
            with_label = np.in1d(labels, map_ids)
            # print(map_name)
            # print("Number of mapped:", with_label.sum())
            table[:, col_offset] = with_label
            col_offset += 1

    write_csv(out_path, table, col_names)


def muscle_attributes(muscle_path, key_muscle,
                      seg_path, key_seg,
                      tmp_folder, target, max_jobs):
    muscle_labels = node_labels(seg_path, key_seg,
                                muscle_path, key_muscle,
                                'muscle', tmp_folder,
                                target, max_jobs, max_overlap=False)

    foreground_id = 255

    # we count everything that has at least 25 % overlap as muscle
    overlap_threshold = .25
    muscle_labels = normalize_overlap_dict(muscle_labels)
    label_ids = np.array([k for k in sorted(muscle_labels.keys())])
    overlap_values = np.array([muscle_labels[label_id].get(foreground_id, 0.)
                               for label_id in label_ids])
    overlap_labels = label_ids[overlap_values > overlap_threshold]

    n_labels = int(label_ids.max()) + 1
    muscle_labels = np.zeros(n_labels, dtype='uint8')
    muscle_labels[overlap_labels] = foreground_id

    semantic_muscle = {'muscle': [foreground_id]}
    return muscle_labels, semantic_muscle


def region_attributes(seg_path, region_out, segmentation_folder,
                      label_ids, tmp_folder, target, max_jobs,
                      key_seg=None):
    if seg_path.endswith('.n5') and key_seg is None:
        key_seg = 'setup0/timepoint0/s2'
    elif key_seg is None:
        key_seg = 't00000/s00/2/cells'

    # 1.) compute the mapping to carved regions
    carved_path = os.path.join(segmentation_folder,
                               'sbem-6dpf-1-whole-segmented-tissue.xml')
    carved_key = get_seg_key_xml(carved_path, scale=0)
    carved_path = get_data_path(carved_path, return_absolute_path=True)
    carved_labels = node_labels(seg_path, key_seg,
                                carved_path, carved_key,
                                'carved-regions', tmp_folder,
                                target, max_jobs)
    # load the mapping of ids to semantics
    with open_file(carved_path, 'r') as f:
        attrs = f.attrs
        names = attrs['semantic_names']
        ids = attrs['semantic_mapping']
    semantics_to_carved_ids = {name: idx for name, idx in zip(names, ids)}
    label_list = [carved_labels]
    semantic_mapping_list = [semantics_to_carved_ids]

    # 2.) compute the mapping to muscles
    muscle_path = os.path.join(segmentation_folder, 'sbem-6dpf-1-whole-segmented-muscle.xml')
    muscle_key = get_seg_key_xml(muscle_path, scale=0)
    muscle_path = get_data_path(muscle_path, return_absolute_path=True)
    # need to be more lenient with the overlap criterion for the muscle mapping
    muscle_labels, semantic_muscle = muscle_attributes(muscle_path, muscle_key,
                                                       seg_path, key_seg,
                                                       tmp_folder, target, max_jobs)
    label_list.append(muscle_labels)
    semantic_mapping_list.append(semantic_muscle)

    # 3.) map all the segmented prospr regions
    region_paths = glob.glob(os.path.join(segmentation_folder,
                                          "prospr-6dpf-1-whole-segmented-*"))
    region_names = [os.path.splitext(pp.split('-')[-1])[0].lower() for pp in region_paths]
    region_keys = [get_seg_key_xml(rpath, scale=0) for rpath in region_paths]
    region_paths = [get_data_path(rp, return_absolute_path=True)
                    for rp in region_paths]
    for rpath, rkey, rname in zip(region_paths, region_keys, region_names):
        rlabels = node_labels(seg_path, key_seg,
                              rpath, rkey,
                              rname, tmp_folder,
                              target, max_jobs)
        label_list.append(rlabels)
        semantic_mapping_list.append({rname: [255]})

    # 4.) map the midgut segmentation
    midgut_path = os.path.join(segmentation_folder, 'sbem-6dpf-1-whole-segmented-midgut.xml')
    midgut_key = get_seg_key_xml(midgut_path, scale=0)
    midgut_path = get_data_path(midgut_path, return_absolute_path=True)
    midgut_labels = node_labels(seg_path, key_seg, midgut_path, midgut_key,
                                'midgut', tmp_folder, target, max_jobs)
    label_list.append(midgut_labels)
    semantic_mapping_list.append({'midgut': [255]})

    # 5.) merge the mappings and write new table
    write_region_table(label_ids, label_list, semantic_mapping_list, region_out)

    # 6.) add nephridia to the table
    nephridia_path = os.path.join(segmentation_folder, 'sbem-6dpf-1-whole-segmented-nephridia.xml')
    nephridia_key = get_seg_key_xml(nephridia_path, scale=0)
    nephridia_path = get_data_path(nephridia_path, return_absolute_path=True)
    nephridia_labels = node_labels(seg_path, key_seg, nephridia_path, nephridia_key,
                                   'nephridia', tmp_folder, target, max_jobs)

    region_table = pd.read_csv(region_out, sep='\t')
    if 'nephridia' in region_table.columns:
        return
    assert len(nephridia_labels) == len(label_ids)
    region_table['nephridia'] = nephridia_labels
    region_table.to_csv(region_out, sep='\t', index=False)


def extrapolated_intensities(seg_path, seg_key, mask_path, mask_key, out_path,
                             tmp_folder, target, max_jobs, overlap_threshold=.5):
    foreground_id = 255
    mask_labels = node_labels(seg_path, seg_key,
                              mask_path, mask_key,
                              'extrapolated_intensities', tmp_folder,
                              target, max_jobs, max_overlap=False)

    # we count everything that has at least 25 % overlap as muscle
    mask_labels = normalize_overlap_dict(mask_labels)
    label_ids = np.array([k for k in sorted(mask_labels.keys())])
    overlap_values = np.array([mask_labels[label_id].get(foreground_id, 0.)
                               for label_id in label_ids])
    overlap_labels = label_ids[overlap_values > overlap_threshold]

    n_labels = int(label_ids.max()) + 1
    mask_labels = np.zeros(n_labels, dtype='uint8')
    mask_labels[overlap_labels] = 1

    label_ids = np.arange(n_labels)
    data = np.concatenate([label_ids[:, None], mask_labels[:, None]], axis=1)
    cols = ['label_id', 'has_extrapolated_intensities']
    table = pd.DataFrame(data, columns=cols)
    table.to_csv(out_path, sep='\t', index=False)
