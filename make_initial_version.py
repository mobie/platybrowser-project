#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
from shutil import copyfile

from scripts.files import make_folder_structure
from scripts.export import export_segmentation
from scripts.files import copy_xml_with_abspath, write_simple_xml
from scripts.files import copy_files_with_pattern
from scripts.attributes import make_nucleus_tables, make_cell_tables


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
    export_segmentation(path, key_cells, old_folder, folder, cells_name, res_cells, tmp_cells,
                        target='local', max_jobs=8)


def copy_aux_gene_file(out_folder):
    # TODO we need some central h5 data storage and put this there
    input_path = '/g/kreshuk/zinchenk/cell_match/data/meds_all_genes_500nm.h5'
    xml_path = os.path.join(out_folder, 'meds_all_genes.xml')
    write_simple_xml(xml_path, input_path)


def make_image_data(old_folder, folder):
    data_folder = os.path.join(folder, 'images')

    # start by copying the raw data
    raw_name = 'em-raw-full-res.xml'
    raw_in = os.path.join(old_folder, raw_name)
    raw_out = os.path.join(data_folder, raw_name)
    copy_xml_with_abspath(raw_in, raw_out)

    # TODO sub-folder ?
    # copy MEDs and SPMs
    copy_files_with_pattern(old_folder, data_folder, "*-MED*")
    # TODO do we need the SPMs here?
    # copy_files_with_pattern(old_folder, data_folder, "*-SPM*")

    # copy valentyna's aux gene file
    misc_folder = os.path.join(folder, 'misc')
    copy_aux_gene_file(misc_folder)

    # copy cellular models
    # TODO

    # copy additional segmentations
    # (muscle, tissue (includes neuropil), TODO more?)
    seg_folder = os.path.join(folder, 'segmentations')
    seg_in_names = ['em-segmented-muscles-ariadne.xml',
                    # 'em-segmented-neuropil-ariadne.xml',
                    'em-segmented-tissue-labels.xml']
    seg_out_names = ['em-segmented-muscles.xml',
                     # 'em-segmented-muscles.xml',
                     'em-segmented-tissue-labels.xml']
    for in_name, out_name in zip(seg_in_names, seg_out_names):
        seg_in = os.path.join(old_folder, in_name)
        seg_out = os.path.join(seg_folder, out_name)
        copy_xml_with_abspath(seg_in, seg_out)

    # also copy the table for the tissue segmentaiton
    table_folder = os.path.join(folder, 'tables', 'em-segmented-tissue-labels')
    os.makedirs(table_folder, exist_ok=True)
    tissue_table_in = os.path.join(old_folder, 'tables', 'em-segmented-tissue-labels.csv')
    tissue_table_out = os.path.join(table_folder, 'base.csv')
    copyfile(tissue_table_in, tissue_table_out)


def make_tables(folder):
    # make cell segmentation tables
    name_cells = 'em-segmented-cells-labels'
    res_cells = [.025, .02, .02]
    make_cell_tables(folder, name_cells, 'tmp_tables_cells',
                     res_cells, target='local', max_jobs=32)

    # make nucleus segmentation tables
    name_nuclei = 'em-segmented-nuclei-labels'
    res_nuclei = [.1, .08, .08]
    make_nucleus_tables(folder, name_nuclei, 'tmp_tables_nuclei',
                        res_nuclei, target='local', max_jobs=32)


def make_initial_version():

    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    tag = '0.0.0'
    folder = os.path.join('data', tag)

    make_folder_structure(folder)
    # make_segmentations(old_folder, folder)

    # make xmls for all necessary image data
    make_image_data(old_folder, folder)

    make_tables(folder)


if __name__ == '__main__':
    make_initial_version()
