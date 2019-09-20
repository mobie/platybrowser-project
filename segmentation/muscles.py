#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python
import json
import numpy as np
import pandas as pd
from scripts.segmentation.muscle import ranked_false_positives, get_mapped_ids
from scripts.segmentation.muscle.muscle_mapping import compute_labels


PROJECT_PATH = '/g/kreshuk/pape/Work/muscle_mapping_v1.h5'
ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.5.2/tables'


def compute_ranked_missing_candidates():
    missing_candidates, mean_distances = ranked_false_positives(ROOT, threshold=.4)
    print(mean_distances.min(), mean_distances.max())
    labels, label_ids = compute_labels(ROOT)

    candidates = np.zeros_like(labels)
    candidates[missing_candidates] = 1

    distances = np.zeros_like(labels, dtype='float32')
    distances[missing_candidates] = mean_distances

    table = np.concatenate([label_ids[:, None], candidates[:, None], distances[:, None]], axis=1)
    table = pd.DataFrame(data=table, columns=['label_id', 'missing_candidates', 'mean_muscle_distance'])
    output_path = '../data/0.5.2/tables/sbem-6dpf-1-whole-segmented-cells-labels/missing_muscle_candidates.csv'
    table.to_csv(output_path, sep='\t', index=False)


def write_diff_segs():

    with open('./muscle_corrections/missed.json') as f:
        ids = json.load(f)
    out_path = '../data/0.5.3/segmentations/sbem-6dpf-1-whole-segmented-muscle-missing.xml'
    get_mapped_ids(ids, out_path)

    with open('./muscle_corrections/need_correction.json') as f:
        extra_ids = json.load(f)
    out_path = '../data/0.5.3/segmentations/sbem-6dpf-1-whole-segmented-muscle-missing-full.xml'
    all_ids = ids + extra_ids
    get_mapped_ids(all_ids, out_path)


if __name__ == '__main__':
    # compute_ranked_missing_candidates()
    write_diff_segs()
