#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python
from scripts.segmentation.muscle import predict_muscle_mapping, run_workflow


PROJECT_PATH = '/g/kreshuk/pape/Work/muscle_mapping_v1.h5'
ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.5.1/tables'


def precompute():
    predict_muscle_mapping(ROOT, PROJECT_PATH)


def proofreading():
    run_workflow(PROJECT_PATH)


if __name__ == '__main__':
    # precompute()
    proofreading()
