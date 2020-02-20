#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
from mmpb.segmentation.nuclei import nucleus_segmentation_workflow


def segment_nuclei(path, target, max_jobs, n_threads, stitch_mode):
    tmp_folder = './tmp_segment_nuclei'

    fg_key = 'volumes/nuclei/foreground'
    aff_key = 'volumes/nuclei/affinities'
    mask_out_key = 'volumes/nuclei/foreground_mask'
    out_key = 'volumes/nuclei/segmentation'

    mask_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-inside.n5'
    mask_key = 'setup0/timepoint0/s0'

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-4, 0, 0], [0, -4, 0], [0, 0, -4],
               [-8, 0, 0], [0, -8, 0], [0, 0, -8],
               [-16, 0, 0], [0, -16, 0], [0, 0, -16]]
    nucleus_segmentation_workflow(offsets, path, fg_key, aff_key,
                                  mask_path, mask_key,
                                  out_key, mask_out_key,
                                  tmp_folder=tmp_folder, target=target,
                                  max_jobs=max_jobs, n_threads=n_threads,
                                  stitch_mode=stitch_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=125)
    parser.add_argument('--n_threads', type=int, default=48)
    parser.add_argument('--stitch_mode', type=str, default='unbiased')
    args = parser.parse_args()
    segment_nuclei(args.path, args.target, args.max_jobs,
                   args.n_threads, args.stitch_mode)
