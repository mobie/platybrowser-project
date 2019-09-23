#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from subprocess import check_output
import luigi

from scripts.files import copy_release_folder, make_folder_structure, make_bdv_server_file
from scripts.release_helper import add_version
from scripts.extension.registration import ApplyRegistrationLocal, ApplyRegistrationSlurm
from scripts.default_config import get_default_shebang


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def apply_registration(input_folder, new_folder,
                       transformation_file, source_prefix,
                       target, max_jobs):
    task = ApplyRegistrationSlurm if target == 'slurm' else ApplyRegistrationLocal
    tmp_folder = './tmp_registration'
    os.makedirs(tmp_folder, exist_ok=True)

    # find all input files
    names = os.listdir(input_folder)
    inputs = [os.path.join(input_folder, name) for name in names]

    if len(inputs) == 0:
        raise RuntimeError("Did not find any files with prefix %s in %s" % (source_prefix,
                                                                            input_folder))

    output_folder = os.path.join(new_folder, 'images')
    # TODO parse names to get output names
    output_names = []
    outputs = [os.path.join(output_folder, name) for name in output_names]

    # update the task config
    config_dir = os.path.join(tmp_folder, 'config')
    os.makedirs(config_dir, exist_ok=True)

    shebang = get_default_shebang()
    global_config = task.default_global_config()
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    # TODO more time than 3 hrs?
    task_config = task.default_task_config()
    task_config.update({'mem_limit': 16, 'time_limit': 180})
    with open(os.path.join(config_dir, 'apply_registration.config'), 'w') as f:
        json.dump(task_config, f)

    # write path name files to json
    input_file = os.path.join(tmp_folder, 'input_files.json')
    with open(input_file, 'w') as f:
        json.dump(inputs, f)
    output_file = os.path.joout(tmp_folder, 'output_files.json')
    with open(output_file, 'w') as f:
        json.dump(outputs, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
             input_path_file=input_file, output_path_file=output_file,
             transformation_file=transformation_file)

    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Registration failed")


def update_regestration(transformation_file, input_folder, source_prefix, target, max_jobs):
    """ Update the prospr segmentation.
    This is a special case of 'update_patch', that applies a new prospr registration.

    Arguments:
        transformation_file [str] - path to the transformation used to register
        input_folder [str] - folder with unregistered data
        source_prefix [str] - prefix of the source data to apply the registration to
        target [str] - target of computation
        max_jobs [int] - max number of jobs for computation
    """
    tag, new_tag = get_tags()
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy the release folder
    copy_release_folder(folder, new_folder)

    # apply new registration to all files of the source prefix
    apply_registration(input_folder, new_folder,
                       transformation_file, source_prefix,
                       target, max_jobs)
    add_version(new_tag)
    make_bdv_server_file(new_folder, os.path.join(new_folder, 'misc', 'bdv_server.txt'),
                         relative_paths=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update prospr registration in platy-browser-data.')
    parser.add_argument('transformation_file', type=str, help="path to transformation file")

    parser.add_argument('--input_folder', type=str, default="data/rawdata/prospr",
                        help="Folder with (not registered) input files")
    help_str = "Prefix for the input data. Please change this if you change the 'input_folder' from its default value"
    parser.add_argument('--source_prefix', type=str, default="prospr-6dpf-1-whole",
                        help=help_str)

    parser.add_argument('--target', type=str, default='slurm',
                        help="Computatin plaform, can be 'slurm' or 'local'")
    parser.add_argument('--max_jobs', type=int, default=100,
                        help="Maximal number of jobs used for computation")

    args = parser.parse_args()
    update_regestration(args.transformation_file, args.input_folder, args.source_prefix,
                        args.target, args.max_jobs)
