import argparse
import os
import numpy as np
import h5py
from scripts.segmentation.validation import eval_cells, get_ignore_seg_ids
from scripts.attributes.region_attributes import region_attributes
from scripts.default_config import write_default_global_config

ANNOTATIONS = '../../data/rawdata/evaluation/validation_annotations.h5'
BASELINES = '../../data/rawdata/evaluation/baseline_cell_segmentations.h5'


def get_label_ids(path, key):
    with h5py.File(path, 'r') as f:
        ds = f[key]
        max_id = ds.attrs['maxId']
    label_ids = np.arange(max_id + 1)
    return label_ids


def compute_baseline_tables():
    names = ['lmc', 'mc', 'curated_lmc', 'curated_mc']
    path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/evaluation',
                        'baseline_cell_segmentations.h5')
    table_prefix = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/evaluation'
    im_folder = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.0/images'
    seg_folder = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.0/segmentations'
    for name in names:
        key = name
        out_path = os.path.join(table_prefix, '%s.csv' % name)
        tmp_folder = './tmp_regions_%s' % name
        config_folder = os.path.join(tmp_folder, 'configs')
        write_default_global_config(config_folder)
        label_ids = get_label_ids(path, key)
        region_attributes(path, out_path, im_folder, seg_folder,
                          label_ids, tmp_folder, target='local', max_jobs=64,
                          key_seg=key)


def eval_seg(path, key, table):
    ignore_ids = get_ignore_seg_ids(table)
    fm, fs, tot = eval_cells(path, key, ANNOTATIONS,
                             ignore_seg_ids=ignore_ids)
    print("Evaluation yields:")
    print("False merges:", fm)
    print("False splits:", fs)
    print("Total number of annotations:", tot)


def eval_baselines():
    path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/evaluation',
                        'baseline_cell_segmentations.h5')
    names = ['lmc', 'mc', 'curated_lmc', 'curated_mc']
    table_prefix = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/evaluation'
    results = {}
    for name in names:
        print("Run evaluation for %s ..." % name)
        table = os.path.join(table_prefix, '%s.csv' % name)
        ignore_ids = get_ignore_seg_ids(table)
        key = name
        fm, fs, tot = eval_cells(path, key, ANNOTATIONS,
                                 ignore_seg_ids=ignore_ids)
        results[name] = (fm, fs, tot)

    for name in names:
        print("Evaluation of", name, "yields:")
        print("False merges:", fm)
        print("False splits:", fs)
        print("Total number of annotations:", tot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to segmentation that should be validated.")
    parser.add_argument("table", type=str, help="Path to table with region/semantic assignments")
    parser.add_argument("--key", type=str, default="t00000/s00/0/cells", help="Segmentation key")
    parser.add_argument("--baselines", type=int, default=0,
                        help="Whether to evaluate the baseline segmentations (overrides path)")
    args = parser.parse_args()

    baselines = bool(args.baselines)
    if baselines:
        eval_baselines()
    else:
        path = args.path
        table = args.table
        key = args.key
        assert os.path.exists(path), path
        eval_seg(path, key, table)


if __name__ == '__main__':
    # compute_baseline_tables()
    main()
