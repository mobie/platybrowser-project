import os
import json
import numpy as np
import z5py
from scripts.segmentation.correction import preprocess
from scripts.segmentation.correction.heuristics import (compute_ratios,
                                                        components_per_slice,
                                                        get_ignore_ids)


def run_preprocessing():
    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    key = 'volumes/paintera/proofread_cells'
    aff_key = 'volumes/curated_affinities/s1'

    out_path = './data.n5'
    out_key = 'segmentation'

    tissue_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                               'rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.h5')
    tissue_key = 't00000/s00/0/cells'

    preprocess(path, key, aff_key, tissue_path, tissue_key,
               out_path, out_key)


def morphology_outlier():
    scale = 2
    p1 = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/analysis/correction/data.n5'
    seg_key = 'segmentation/s%i' % scale
    table_key = 'morphology'
    scale_factor = 2 ** scale

    with z5py.File(p1, 'r') as f:
        ds = f['segmentation/s0']
        n_ids = ds.attrs['maxId'] + 1

    seg_ids = np.arange(n_ids)
    ignore_ids = get_ignore_ids(p1, 'tissue_labels')
    ignore_id_mask = ~np.isin(seg_ids, ignore_ids)
    seg_ids = seg_ids[ignore_id_mask]

    seg_ids, ratios_close = compute_ratios(p1, seg_key, p1, table_key, scale_factor,
                                           64, components_per_slice, seg_ids, True)
    with open('./ratios_components.json', 'w') as f:
        json.dump(ratios_close.tolist(), f)
    with open('./ids_morphology.json', 'w') as f:
        json.dump(seg_ids.tolist(), f)


if __name__ == '__main__':
    # run_preprocessing()
    morphology_outlier()
