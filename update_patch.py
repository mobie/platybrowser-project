#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from copy import deepcopy
from glob import glob
from shutil import rmtree
from subprocess import check_output, call

import scripts.attributes
from scripts.files import get_segmentation_names, get_segmentations
from scripts.files import (copy_image_data, copy_misc_data, copy_segmentation, copy_tables,
                           make_bdv_server_file, make_folder_structure)
from scripts.export import export_segmentation
from scripts.release_helper import add_version


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def update_segmentation(name, seg_dict, folder, new_folder,
                        target, max_jobs):
    tmp_folder = 'tmp_export_%s' % name
    paintera_root, paintera_key = seg_dict['paintera_project']
    export_segmentation(paintera_root, paintera_key,
                        folder, new_folder, name,
                        resolution=seg_dict['resolution'],
                        tmp_folder=tmp_folder,
                        target=target, max_jobs=max_jobs)


def update_segmentations(folder, new_folder, names_to_update, target, max_jobs):
    segmentations = get_segmentations()
    segmentation_names = get_segmentation_names()

    for name in segmentation_names:
        if name in names_to_update:
            update_segmentation(name, segmentations[name], folder, new_folder,
                                target, max_jobs)
        else:
            copy_segmentation(folder, new_folder, name)


def update_table(name, seg_dict, folder, new_folder,
                 target, max_jobs):
    tmp_folder = 'tmp_tables_%s' % name
    update_function = getattr(scripts.attributes, seg_dict['table_update_function'])
    update_function(folder, new_folder, name, tmp_folder, seg_dict['resolution'],
                    target=target, max_jobs=max_jobs)


def update_tables(folder, new_folder, names_to_update, target, max_jobs):
    segmentations = get_segmentations()

    # first copy all tables that just need to be copied
    for name, seg in segmentations.items():
        has_table = seg.get('has_tables', False) or 'table_update_function' in seg
        needs_update = name in names_to_update
        if not has_table or needs_update:
            continue
        copy_tables(folder, new_folder, name)

    # now update all tables that need to be updated
    for name, seg in segmentations.items():
        has_table = seg.get('has_tables', False) or 'table_update_function' in seg
        needs_update = name in names_to_update
        if not needs_update:
            continue
        if needs_update and not has_table:
            raise RuntimeError("Segmentation %s does not have a table:" % name)
        update_table(name, seg, folder, new_folder,
                     target, max_jobs)


# TODO check for errors
def make_release(tag, folder, description=''):
    call(['git', 'add', folder])
    call(['git', 'commit', '-m', 'Automatic platybrowser update'])
    if description == '':
        call(['git', 'tag', tag])
    else:
        call(['git', '-m', description, 'tag', tag])
    # TODO use the gitlab api instead
    # call(['git', 'push', 'origin', 'master', '--tags'])


def clean_up():
    """ Clean up all tmp folders
    """

    def remove_dir(dir_name):
        try:
            rmtree(dir_name)
        except OSError:
            pass

    tmp_folders = glob('tmp_*')
    for tmp_folder in tmp_folders:
        remove_dir(tmp_folder)


def check_requested_updates(names_to_update):
    segmentations = get_segmentations()
    for name in names_to_update:
        if name not in segmentations:
            raise ValueError("Requested update for %s, which is not a registered segmentation" % name)
        if segmentations[name]['is_static']:
            raise ValueError("Requested update for %s, which is a static segmentation" % name)


# TODO catch all exceptions and handle them properly
def update_patch(update_seg_names, update_table_names,
                 description='', target='slurm', max_jobs=250):
    """ Generate new patch version of platy-browser derived data.

    The patch version is increased if derived data changes, e.g. by
    incorporating corrections for a segmentation or updating tables.

    Arguments:
        update_seg_names [list[str]] - names of segmentations to be updated.
        update_table_names [list[str]] - names of tables to be updated.
            Not that these only need to be specified if the corresponding segmentation is not
            updated, but the tables should be updated.
        description [str] - optional descrption for release message (default: '').
        target [str] - target for the computation ('local' or 'slurm', default is 'slurm').
        max_jobs [int] - maximal number of jobs used for computation (default: 250).
    """

    # check if we have anything to update
    have_seg_updates = len(update_seg_names) > 0
    have_table_updates = len(update_seg_names) > 0
    if not have_seg_updates and not have_table_updates:
        raise ValueError("No updates where provdied and force_update was not set")

    table_updates = deepcopy(update_seg_names)
    table_updates.extend(update_table_names)
    check_requested_updates(table_updates)

    # increase the patch (last digit) release tag
    tag, new_tag = get_tags()
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy static image and misc data
    copy_image_data(folder, new_folder)
    copy_misc_data(folder, new_folder)

    # export new segmentations
    update_segmentations(folder, new_folder, update_seg_names,
                         target=target, max_jobs=max_jobs)

    # generate new attribute tables
    update_tables(folder, new_folder, table_updates,
                  target=target, max_jobs=max_jobs)

    print("All updates were successfull. Making bdv file and adding version tag")
    make_bdv_server_file(new_folder, os.path.join(new_folder, 'misc', 'bdv_server.txt'),
                         relative_paths=True)
    add_version(new_tag)
    # TODO add some quality control that cheks that all files are there

    # TODO implement make release properly
    return
    # make new release
    make_release(new_tag, new_folder, description)
    print("Updated platybrowser to new release", new_tag)
    print("All changes were successfully made. Starting clean up now.")
    print("This can take a few hours, you can already use the new data.")
    print("Clean-up will only remove temp files.")
    clean_up()


if __name__ == '__main__':
    help_str = "Path to a json containing list of the data to update. See docstring of 'update_patch' for details."
    parser = argparse.ArgumentParser(description='Update patch version of platy-browser-data.')
    parser.add_argument('input_path', type=str, help=help_str)
    parser.add_argument('--target', type=str, default='slurm',
                        help="Computatin plaform, can be 'slurm' or 'local'")
    parser.add_argument('--max_jobs', type=int, default=250,
                        help="Maximal number of jobs used for computation")
    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path) as f:
        update_dict = json.load(f)
    if not ('segmentations' in update_dict and 'tables' in update_dict):
        raise ValueError("Invalid udpate file")
    update_patch(update_dict['segmentations'], update_dict['tables'],
                 target=args.target, max_jobs=args.max_jobs)
