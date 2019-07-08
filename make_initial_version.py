#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
from shutil import copyfile
from glob import glob

import h5py
from scripts.files import make_folder_structure
from scripts.export import export_segmentation
from scripts.files import make_bdv_server_file, copy_image_data, copy_misc_data
from scripts.attributes import make_nucleus_tables, make_cell_tables
from pybdv.converter import make_bdv


def make_sbem_segmentations(old_folder, folder):
    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'

    # export nucleus segemntation
    tmp_nuclei = 'tmp_export_nuclei'
    key_nuclei = 'volumes/paintera/nuclei'
    nuclei_name = 'sbem-6dpf-1-whole-segmented-nuclei-labels'
    res_nuclei = [.1, .08, .08]
    export_segmentation(path, key_nuclei, old_folder, folder, nuclei_name, res_nuclei, tmp_nuclei)

    # export cell segemntation
    tmp_cells = 'tmp_export_cells'
    key_cells = 'volumes/paintera/proofread_cells'
    cells_name = 'sbem-6dpf-1-whole-segmented-cells-labels'
    res_cells = [.025, .02, .02]
    export_segmentation(path, key_cells, old_folder, folder, cells_name, res_cells, tmp_cells,
                        target='local', max_jobs=8)


def make_sbem_tables(folder):
    # make cell segmentation tables
    name_cells = 'sbem-6dpf-1-whole-segmented-cells-labels'
    res_cells = [.025, .02, .02]
    make_cell_tables(folder, name_cells, 'tmp_tables_cells',
                     res_cells, target='local', max_jobs=32)

    # make nucleus segmentation tables
    name_nuclei = 'sbem-6dpf-1-whole-segmented-nuclei-labels'
    res_nuclei = [.1, .08, .08]
    make_nucleus_tables(folder, name_nuclei, 'tmp_tables_nuclei',
                        res_nuclei, target='local', max_jobs=32)

    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    # copy tissue segmentation table
    tissue_name_out = 'sbem-6dpf-1-whole-segmented-tissue-labels'
    table_folder = os.path.join(folder, 'tables', tissue_name_out)
    os.makedirs(table_folder, exist_ok=True)
    tissue_table_in = os.path.join(old_folder, 'tables', 'em-segmented-tissue-labels.csv')
    tissue_table_out = os.path.join(table_folder, 'base.csv')
    copyfile(tissue_table_in, tissue_table_out)


def make_prospr_region_segmentations():
    in_prefix = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/BodyPart_*.h5'
    out_prefix = './data/rawdata/prospr-6dpf-1-whole-segmented-'
    files = glob(in_prefix)
    for p in files:
        name = p.split('_')[-1][:-3]
        o = out_prefix + name + '.h5'
        print(p, "to", o)
        with h5py.File(p) as f:
            key = 't00000/s00/0/cells'
            data = f[key][:]
            data[data > 0] = 0
            data[data < 0] = 255
        make_bdv(data, o, 3 * [[2, 2, 2]],
                 unit='micrometer', resolution=[0.5, 0.5, 0.5])


def make_initial_version():

    src_folder = 'data/rawdata'
    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    tag = '0.0.0'
    folder = os.path.join('data', tag)

    make_folder_structure(folder)

    # make xmls for all necessary image data
    copy_image_data(src_folder, os.path.join(folder, 'images'))
    copy_misc_data(src_folder, os.path.join(folder, 'misc'))

    # export the initial sbem segmentations
    make_sbem_segmentations(old_folder, folder)

    # make the tables for sbem segmentations
    make_sbem_tables(folder)

    # make the bdv server file
    make_bdv_server_file([os.path.join(folder, 'images'),
                          os.path.join(folder, 'segmentations')],
                         os.path.join(folder, 'misc', 'bdvserver.txt'))



if __name__ == '__main__':
    # make_prospr_region_segmentations()
    make_initial_version()
