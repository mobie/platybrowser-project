# TODO shebang
import argparse
import os
from scripts.segmentation.validation import eval_cells, get_ignore_seg_ids

ANNOTATIONS = '../../data/rawdata/evaluation/validation_annotations.h5'
BASELINES = '../../data/rawdata/evaluation/baseline_cell_segmentations.h5'


def eval_seg(path, key, table):
    ignore_ids = get_ignore_seg_ids(table)
    fm, fs, tot = eval_cells(path, key, ANNOTATIONS,
                             ignore_seg_ids=ignore_ids)
    print("Evaluation yields:")
    print("False merges:", fm)
    print("False splits:", fs)
    print("Total number of annotations:", tot)


def eval_baselines():
    names = ['lmc', 'mc', 'curated_lmc', 'curated_mc']
    # TODO still need to compute region tables for the baselines
    tables = ['',
              '',
              '',
              '']
    results = {}
    for name, table in zip(names, tables):
        ignore_ids = get_ignore_seg_ids(table)
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
    parse.add_argument("--key", type=str, default="t00000/s00/0/cells", help="Segmentation key")
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
    main()
