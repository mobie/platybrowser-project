#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
from mmpb.segmentation.cells import curate_affinities


def get_roi(path, key, halo=[100, 1024, 1024]):
    with open_file(path, 'r') as f:
        shape = f[key].shape[1:]
    roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    return roi_begin, roi_end


def run_curation(path, target, max_jobs, with_roi):
    tmp_folder = './tmp_curate_affinities'
    in_key = 'volumes/cells/affinities/s1'
    out_key = 'volumes/cells/curated_affinities/s1'

    region_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-tissue.n5'
    region_key = 'setup0/timepoint0/s0'

    if with_roi:
        roi_begin, roi_end = get_roi(path, in_key)
    else:
        roi_begin = roi_end = None

    run_curation(path, in_key, out_key,
                 region_path, region_key,
                 tmp_folder, target, max_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=400)
    parser.add_argument('--with_roi', type=int, default=0)
    curate_affinities(args.path, args.target, args.max_jobs, args.with_roi)
