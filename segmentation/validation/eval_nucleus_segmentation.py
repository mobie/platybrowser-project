import argparse
import os
from mmpb.release_helper import get_version
from mmpb.segmentation.validation import eval_nuclei
from pybdv.metadata import get_data_path

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
ANNOTATIONS = os.path.join(ROOT, 'rawdata/evaluation/validation_annotations.h5')
NAME = 'sbem-6dpf-1-whole-segmented-nuclei'


def eval_seg(version):
    seg_path = os.path.join(ROOT, version, 'images', 'local', NAME + '.xml')
    seg_path = get_data_path(seg_path, return_absolute_path=True)
    if seg_path.endswith('.n5'):
        key = 'setup0/timepoint0/s0'
    else:
        key = 't00000/s00/0/cells'

    fp, fn, tot = eval_nuclei(seg_path, key, ANNOTATIONS)
    print("Evaluation yields:")
    print("False positives:", fp)
    print("False negatives:", fn)
    print("Total number of annotations:", tot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='', help="Version to evaluate.")
    args = parser.parse_args()

    version = args.version
    if version == '':
        version = get_version(ROOT)

    eval_seg(version)


if __name__ == '__main__':
    main()
