#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse

from mmpb.bookmarks import update_bookmarks
from mmpb.files import copy_and_check_image_dict, copy_release_folder
from mmpb.release_helper import (add_data, add_version, get_modality_names,
                                 get_version, make_folder_structure)


def get_tags():
    tag = get_version()
    new_tag = tag.split('.')
    new_tag[-1] = '0'  # reset patch
    new_tag[1] = '0'  # reset minor
    # update major
    new_tag[0] = str(int(new_tag[0]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def update_major(new_data, bookmarks=None, target='slurm', max_jobs=250):
    """ Update major version of platy browser.

    The major version is increased if a new primary data source is added.

    Arguments:
        new_data [dict] - new data to be added.
            For details, see https://github.com/platybrowser/platybrowser-backend#table-storage.
        bookmarks [dict] - bookmarks to be added (default: None)
        target [str] - target for the computation ('local' or 'slurm', default is 'slurm').
        max_jobs [int] - maximal number of jobs used for computation (default: 250).
    """

    # increase the major (first digit) release tag
    tag, new_tag = get_tags()
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy the release folder
    copy_release_folder(folder, new_folder)

    # copy image dict and check that all image and table files are there
    copy_and_check_image_dict(folder, new_folder)

    # updated bookmarks if given
    if bookmarks is not None:
        update_bookmarks(new_folder, bookmarks)

    # validate add the new data
    modality_names = get_modality_names('data', tag)
    for name, properties in new_data.items():
        # validate that the name is in the existing modalities
        modality = '-'.join(name.split('-')[:4])
        if modality in modality_names:
            raise ValueError("Expect new modality but got existing one: %s" % modality)
        add_data(name, properties, new_folder, target, max_jobs)

    add_version(new_tag)
    print("Updated platybrowser to new release", new_tag)


if __name__ == '__main__':
    help_str = "Path to a json containing list of the data to add. See docstring of 'update_major' for details."
    parser = argparse.ArgumentParser(description='Update major version of platy-browser-data.')
    parser.add_argument('input_path', type=str, help=help_str)
    parser.add_argument('--target', type=str, default='slurm',
                        help="Computatin plaform, can be 'slurm' or 'local'")
    parser.add_argument('--max_jobs', type=int, default=250,
                        help="Maximal number of jobs used for computation")

    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path) as f:
        new_data = json.load(f)

    bookmarks = new_data.pop('bookmarks', None)
    update_major(new_data, bookmarks, target=args.target, max_jobs=args.max_jobs)
