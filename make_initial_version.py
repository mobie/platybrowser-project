#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os

from scripts.files import make_folder_structure
from scripts.export import export_segmentation


def make_segmentations(old_folder, folder):
    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'

    # export nucleus segemntation
    # key_nuclei = 'volumes/paintera/nuclei'
    # nuclei_name = 'em-segmented-nuclei-labels'
    # res_nuclei = [.1, .08, .08]
    # export_segmentation(path, key_nuclei, old_folder, folder, nuclei_name, res_nuclei)

    # export cell segemntation
    key_cells = 'volumes/paintera/proofread_cells'
    cells_name = 'em-segmented-cells-labels'
    res_cells = [.025, .02, .02]
    export_segmentation(path, key_cells, old_folder, folder, cells_name, res_cells)


def make_initial_version():

    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    tag = '0.0.0'
    folder = os.path.join('data', tag)

    # make_folder_structure(folder)

    make_segmentations(old_folder, folder)

    # TODO make xmls for all necessary image data

    # TODO make tables


if __name__ == '__main__':
    make_initial_version()
