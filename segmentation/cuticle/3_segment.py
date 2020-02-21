#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
from mmpb.segmentation.cuticle import cuticle_segmentation_workflow


def segment_cuticle(path, target, max_jobs, n_threads):
    tmp_folder = './tmp_segment_cuticle'

    fg_key = 'volumes/cuticle/foreground'
    aff_key = 'volumes/cuticle/affinities'
    mask_out_key = 'volumes/cuticle/foreground_mask'
    out_key = 'volumes/cuticle/segmentation'

    mask_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-shell.n5'
    mask_key = 'setup0/timepoint0/s0'

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-4, 0, 0], [0, -4, 0], [0, 0, -4],
               [-8, 0, 0], [0, -8, 0], [0, 0, -8],
               [-16, 0, 0], [0, -16, 0], [0, 0, -16]]
    cuticle_segmentation_workflow(offsets, path,
                                  fg_key, aff_key, mask_out_key, out_key,
                                  mask_path, mask_key,
                                  tmp_folder=tmp_folder, target=target,
                                  max_jobs=max_jobs, n_threads=n_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=125)
    parser.add_argument('--n_threads', type=int, default=48)
    args = parser.parse_args()
    segment_cuticle(args.path, args.target, args.max_jobs, args.n_threads)
