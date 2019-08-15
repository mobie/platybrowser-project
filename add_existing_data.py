#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python

import os
from glob import glob
from scripts.files import add_source, add_image, add_segmentation


def add_sources():
    # add em source
    add_source('sbem', '6dpf')
    # add prospr source
    add_source('prospr', '6dpf')


def add_images():
    base_folder = './data/0.2.1/images'

    # add sbem raw data
    sbem_prefix = 'sbem-6dpf-1-whole'
    sbem_raw = './data/0.2.1/images/sbem-6dpf-1-whole-raw.xml'
    name = 'raw'
    add_image(sbem_prefix, name, sbem_raw, copy_data=False)

    # add all prospr images
    prospr_prefix = 'prospr-6dpf-1-whole'
    prospr_ims = glob(os.path.join(base_folder, 'prospr-6dpf-1-whole-*'))
    for impath in prospr_ims:
        name = os.path.split(impath)[1]
        name, ext = os.path.splitext(name)
        if ext != '.xml':
            continue
        name = name[(len(prospr_prefix) + 1):]
        add_image(prospr_prefix, name, impath, copy_data=False)


def add_static_segmentations():
    source = 'sbem-6dpf-1-whole'

    # chromatin segmentation
    # chromatin_tables = {'default': './data/0.2.0/tables/sbem-6dpf-1-whole-segmented-chromatin-labels/default.csv'}
    # add_segmentation(source, 'segmented-chromatin-labels',
    #                  segmentation_path='./data/rawdata/sbem-6dpf-1-whole-segmented-chromatin-labels.xml',
    #                  table_path_dict=chromatin_tables, copy_data=False)

    # tissue segmentation
    tissue_tables = {'default': './data/0.1.0/tables/sbem-6dpf-1-whole-segmented-tissue-labels/base.csv'}
    add_segmentation(source, 'segmented-tissue-labels',
                     segmentation_path='./data/rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.xml',
                     table_path_dict=tissue_tables, copy_data=False)

    # muscle segmentation
    add_segmentation(source, 'segmented-muscle',
                     segmentation_path='./data/rawdata/sbem-6dpf-1-whole-segmented-muscle.xml', copy_data=False)


def add_segmentations():
    add_static_segmentations()
    # add_dynamic_segmentations()


def add_existing_data():
    """ Add existing data to the json files that keep track of
        sources, image data and segmentations.
    """
    # add_sources()
    # add_images()
    add_segmentations()


if __name__ == '__main__':
    add_existing_data()
