#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from subprocess import check_output

from scripts.files import (add_image, add_segmentation,
                           copy_release_folder, make_folder_structure)
from scripts.export import export_segmentation


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = '0'  # reset patch
    new_tag[1] = str(int(new_tag[1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def is_image(data):
    if 'source' in data and 'name' in data and 'input_path' in data:
        return True
    return False


def is_static_segmentation(data):
    if 'source' in data and 'name' in data and 'segmentation_path' in data:
        return True
    return False


def is_dynamic_segmentation(data):
    if 'source' in data and 'name' in data and 'paintera_project' in data and 'resolution' in data:
        if len(data['paintera_project']) != 2 or len(data['resolution']) != 3:
            return False
        return True
    return False


def check_inputs(new_data):
    if not all(isinstance(data, dict) for data in new_data):
        raise ValueError("Expect list of dicts as input")

    for data in new_data:
        if not any(is_image(data), is_static_segmentation(data), is_dynamic_segmentation(data)):
            raise ValueError("Could not parse input element %s" % str(data))


def add_data(data, folder, target, max_jobs):
    if is_image(data):
        # register the image data
        add_image(data['source'], data['name'], data['input_path'])

        # copy image data to new release folder

    elif is_static_segmentation(data):
        # register the static segmentation
        add_segmentation(data['source'], data['name'],
                         segmentation_path=data['segmentation_path'],
                         table_path_dict=data.get('table_path_dict', None))

        # copy segmentation data to new release folder

        # if we have tables, copy them as well

    elif is_dynamic_segmentation(data):
        # register the dynamic segmentation
        add_segmentation(data['source'], data['name'],
                         paintera_project=data['paintera_project'],
                         resolution=data['resolution'],
                         table_update_function=data.get('table_update_function', None))

        # export segmentation data to new release folder
        export_segmentation()

        # if we have a table update function, call it


def update_minor(new_data, target='slurm', max_jobs=250):
    """ Update minor version of platy browser.

    TODO explain elements of input list.
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

    # TODO auto-release
    # TODO clean up


if __name__ == '__main__':
    help_str = "Path to a json containing list of the data to add. See docstring of 'update_minor' for details."
    parser = argparse.ArgumentParser(description='Update minor version of platy-browser-data.')
    parser.add_argument('input_path', type=str, help=help_str)
    input_path = parser.parse_args().input_path
    with open(input_path) as f:
        new_data_list = json.load(f)
    update_minor(new_data_list)
