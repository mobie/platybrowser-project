#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from subprocess import check_output

from scripts.files import copy_release_folder, make_folder_structure, make_bdv_server_file
from scripts.release_helper import add_data, check_inputs, add_version


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = '0'  # reset patch
    new_tag[1] = str(int(new_tag[1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def update_minor(new_data, target='slurm', max_jobs=250):
    """ Update minor version of platy browser.

    The minor version is increased if new derived data is added.

    Arguments:
        new_data [list] - list of new data to add. For details, see
            https://git.embl.de/tischer/platy-browser-tables#usage.
        target [str] - target for the computation ('local' or 'slurm', default is 'slurm').
        max_jobs [int] - maximal number of jobs used for computation (default: 250).
    """
    check_inputs(new_data)

    # increase the minor (middle digit) release tag
    tag, new_tag = get_tags()
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy the release folder
    copy_release_folder(folder, new_folder)

    # add the new data
    for data in new_data:
        add_data(data, new_folder, target, max_jobs)

    # make bdv file
    make_bdv_server_file(new_folder, os.path.join(new_folder, 'misc', 'bdv_server.txt'),
                         relative_paths=True)
    add_version(new_tag)

    # TODO auto-release
    # TODO clean up


if __name__ == '__main__':
    help_str = "Path to a json containing list of the data to add. See docstring of 'update_minor' for details."
    parser = argparse.ArgumentParser(description='Update minor version of platy-browser-data.')
    parser.add_argument('input_path', type=str, help=help_str)
    parser.add_argument('--target', type=str, default='slurm',
                        help="Computatin plaform, can be 'slurm' or 'local'")
    parser.add_argument('--max_jobs', type=int, default=250,
                        help="Maximal number of jobs used for computation")
    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path) as f:
        new_data_list = json.load(f)
    update_minor(new_data_list, target=args.target, max_jobs=args.max_jobs)
