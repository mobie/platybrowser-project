#
# just keeping this here for reference
#

import os
import json


def get_skipped_ids(out_path):
    processed_ids = './proj_correct_fms/processed_ids.json'
    res_dir = './proj_correct_fms/results'
    with open(processed_ids) as f:
        processed_ids = json.load(f)

    saved_ids = os.listdir(res_dir)
    saved_ids = [int(os.path.splitext(sid)[0]) for sid in saved_ids]

    skipped_ids = list(set(processed_ids) - set(saved_ids))
    print("N-skipped:")
    print(len(skipped_ids))

    with open(out_path, 'w') as f:
        json.dump(skipped_ids, f)


def check_export():
    from scripts.segmentation.correction.export_node_labels import check_exported_paintera

    path1 = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    assignment_key = 'volumes/paintera/proofread_cells/fragment-segment-assignment'
    raw_key = 'volumes/raw/s3'
    seg_key = 'volumes/paintera/proofread_cells/data/s2'

    path2 = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/analysis/correction/data.n5'
    node_label_key = 'node_labels'
    table_key = 'morphology'
    scale_factor = 4

    res_folder = './proj_correct_fms/results'
    ids = os.listdir(res_folder)[:10]
    ids = [int(os.path.splitext(idd)[0]) for idd in ids]

    check_exported_paintera(path1, assignment_key,
                            path2, node_label_key,
                            path2, table_key, scale_factor,
                            path1, raw_key, path1, seg_key,
                            ids)
