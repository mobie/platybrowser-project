#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python
from scripts.segmentation.cells.multicut import workflow


# TODO need to expose the path options here


def run_workflow():
    target = 'slurm'

    use_curated_affs = False
    use_lmc = True

    workflow(use_curated_affs, use_lmc, target)


if __name__ == '__main__':
    run_workflow()
