#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os

from scripts.files import make_folder_structure
from scripts.export import export_segmentation
from scripts.files import copy_xml_with_abspath
from scripts.attributes import make_nucleus_tables


def make_segmentations(old_folder, folder):
    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'

    # export nucleus segemntation
    tmp_nuclei = 'tmp_export_nuclei'
    key_nuclei = 'volumes/paintera/nuclei'
    nuclei_name = 'em-segmented-nuclei-labels'
    res_nuclei = [.1, .08, .08]
    export_segmentation(path, key_nuclei, old_folder, folder, nuclei_name, res_nuclei, tmp_nuclei)

    # export cell segemntation
    tmp_cells = 'tmp_export_cells'
    key_cells = 'volumes/paintera/proofread_cells'
    cells_name = 'em-segmented-cells-labels'
    res_cells = [.025, .02, .02]
    export_segmentation(path, key_cells, old_folder, folder, cells_name, res_cells, tmp_cells)


def make_image_data(old_folder, folder):
    data_folder = os.path.join(folder, 'images')

    # start by copying the raw data
    raw_name = 'em-raw-full-res.xml'
    raw_in = os.path.join(old_folder, raw_name)
    raw_out = os.path.join(data_folder, raw_name)
    copy_xml_with_abspath(raw_in, raw_out)

    # TODO
    # copy MEDs and SPMs
    # copy cellular models
    # copy additional segmentations from tischi and ariadne
    # (neuropil, muscle, ...)


def make_tables(folder):
    # TODO make cell segmentation tables
    # name_cells = 'em-segmented-cells-labels'
    # res_cells = [.025, .02, .02]
    # make_cell_tables(folder, name_cells, 'tmp_tables_cells', res_cells)

    # make nucleus segmentation tables
    name_nuclei = 'em-segmented-nuclei-labels'
    res_nuclei = [.1, .08, .08]
    make_nucleus_tables(folder, name_nuclei, 'tmp_tables_nuclei',
                        res_nuclei, target='local', max_jobs=8)


def make_initial_version():

    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    tag = '0.0.0'
    folder = os.path.join('data', tag)

    # make_folder_structure(folder)
    # make_segmentations(old_folder, folder)

    # make xmls for all necessary image data
    # make_image_data(old_folder, folder)

    make_tables(folder)


if __name__ == '__main__':
    make_initial_version()
