import argparse
import os
from scripts.segmentation.validation import eval_nuclei

ANNOTATIONS = '../../data/rawdata/evaluation/validation_annotations.h5'


def eval_seg(path, key):
    fp, fn, tot = eval_nuclei(path, key, ANNOTATIONS)
    print("Evaluation yields:")
    print("False positives:", fp)
    print("False negatives:", fn)
    print("Total number of annotations:", tot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to nuclei segmentation that should be validated.")
    parser.add_argument("--key", type=str, default="t00000/s00/0/cells", help="Segmentation key")
    args = parser.parse_args()

    path = args.path
    key = args.key
    assert os.path.exists(path), path
    eval_seg(path, key)


if __name__ == '__main__':
    main()
