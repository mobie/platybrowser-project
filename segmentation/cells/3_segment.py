#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
from mmpb.segmentation.cells.multicut import workflow


# TODO support bounding box!
def run_workflow():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--curated', type=int, default=1)
    parser.add_argument('--lifted', type=int, default=1)
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=400)
    args = parser.parse_args()

    workflow(args.path,
             use_curated_affs=bool(args.curated), use_lmc=bool(args.lifted),
             target=args.target, max_jobs=args,max_jobs)


if __name__ == '__main__':
    run_workflow()
