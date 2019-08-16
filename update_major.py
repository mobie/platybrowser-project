#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from subprocess import check_output

from scripts.files import add_source, copy_release_folder, make_folder_structure, make_bdv_server_file
from scripts.release_helper import add_data, check_inputs


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = '0'  # reset patch
    new_tag[1] = '0'  # reset minor
    # update major
    new_tag[0] = str(int(new_tag[0]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def update_major(new_data_dict, target='slurm', max_jobs=250):
    """ Update major version of platy browser.

    The major version is increased if a new primary data source is added.

    Arguments:
        new_data [dict] - dictionary of new primary data sources and data to be added. For details, see
            https://git.embl.de/tischer/platy-browser-tables#usage.
        target [str] - target for the computation ('local' or 'slurm', default is 'slurm').
        max_jobs [int] - maximal number of jobs used for computation (default: 250).
    """

    for source, new_data in new_data_dict.items():
        add_source(source)
        check_inputs(new_data, check_source=False)

    # increase the minor (middle digit) release tag
    tag, new_tag = get_tags()
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy the release folder
    copy_release_folder(folder, new_folder)

    # add the new sources and new data
    for source, new_data in new_data.items():
        for data in new_data:
            add_data(data, new_folder, target, max_jobs,
                     source=source)

    # make bdv file
    make_bdv_server_file(new_folder, os.path.join(new_folder, 'misc', 'bdv_server.txt'),
                         relative_paths=True)

    # TODO auto-release
    # TODO clean up


if __name__ == '__main__':
    help_str = "Path to a json containing list of the data to add. See docstring of 'update_major' for details."
    parser = argparse.ArgumentParser(description='Update major version of platy-browser-data.')
    parser.add_argument('input_path', type=str, help=help_str)
    input_path = parser.parse_args().input_path
    with open(input_path) as f:
        new_data_dict = json.load(f)
    update_major(new_data_dict)
