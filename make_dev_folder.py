#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python

import argparse
import os
from subprocess import check_output
from scripts.files import copy_release_folder


def make_dev_folder(dev_name, version=''):
    """
    """
    new_folder = os.path.join('data', 'dev-%s' % dev_name)
    if os.path.exists(new_folder):
        raise RuntimeError("Cannot create dev-folder %s because it already exists" % new_folder)

    if version == '':
        version = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    folder = os.path.join('data', version)
    if not os.path.exists(folder):
        raise RuntimeError("Source folder %s does not exist" % folder)

    copy_release_folder(folder, new_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make dev folder of platy-browser-data.')
    parser.add_argument('dev_name', type=str, help="Name of the dev folder. Will be prefixed with 'dev-'.")
    parser.add_argument('--version', type=str, default='',
                        help="Version of the data used as source. By default, the latest version is used.")
    args = parser.parse_args()
    make_dev_folder(args.dev_name, args.version)
