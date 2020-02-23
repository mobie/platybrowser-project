#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
from elf.io import open_file
from mmpb.segmentation.cells import cell_segmentation_workflow


def get_roi(path, key, halo=[100, 1024, 1024]):
    with open_file(path, 'r') as f:
        shape = f[key].shape[1:]
    roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    return roi_begin, roi_end


def segment_cells():
    default_path = '../data.n5'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=default_path)
    parser.add_argument('--aff_path', type=str, default=default_path)
    parser.add_argument('--curated', type=int, default=1)
    parser.add_argument('--lifted', type=int, default=1)
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=400)
    parser.add_argument('--with_roi', type=int, default=0)
    args = parser.parse_args()

    mask_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-inside.n5'
    mask_key = 'setup0/timepoint0/s0'

    region_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-tissue.n5'
    region_key = 'setup0/timepoint0/s0'

    if args.with_roi:
        roi_begin, roi_end = get_roi(args.path, 'volumes/cells/affinities/s1')
    else:
        roi_begin = roi_end = None

    cell_segmentation_workflow(args.path, args.aff_path,
                               mask_path, mask_key,
                               region_path, region_key,
                               bool(args.curated), bool(args.lifted),
                               args.tmp_folder, args.target, args.max_jobs,
                               roi_begin, roi_end)


if __name__ == '__main__':
    segment_cells()
