#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python

import os
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
    update_function = getattr(scripts.attribute, seg_dict['table_update_function'])
    update_function(new_folder, name, tmp_folder, seg_dict['resolution'],
                    target=target, max_jobs=max_jobs)


def update_tables(folder, new_folder, names_to_update, target, max_jobs):
    segmentations = get_segmentations()
    segmentation_names = get_segmentation_names()

    for name in segmentation_names:
        if name in names_to_update:
            update_table(name, segmentations[name], folder, new_folder,
                         target, max_jobs)
        else:
            copy_tables(folder, new_folder, name)


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
                 description='', force_update=False,
                 target='slurm', max_jobs=250):
    """ Generate new patch version of platy-browser derived data.

    The patch version is increased if derived data changes, e.g. by
    incorporating corrections for a segmentation or updating tables.

    Arguments:
        update_seg_names [list[str]] - names of segmentations to be updated.
        update_table_names [list[str]] - names of tables to be updated.
            Not that these only need to be specified if the corresponding segmentation is not
            updated, but the tables should be updated.
        description [str] - Optional descrption for release message (default: '').
        force_update [bool] - Force an update if no changes are specified (default: False).
        target [str] -
        max_jobs [int] -
    """

    # check if we have anything to update
    have_seg_updates = len(update_seg_names) > 0
    have_table_updates = len(update_seg_names) > 0
    if not have_seg_updates and not have_table_updates and not force_update:
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
    copy_image_data(os.path.join(folder, 'images'),
                    os.path.join(new_folder, 'images'))
    copy_misc_data(os.path.join(folder, 'misc'),
                   os.path.join(new_folder, 'misc'))

    # export new segmentations
    update_segmentations(folder, new_folder, update_seg_names,
                         target=target, max_jobs=max_jobs)

    # generate new attribute tables
    update_tables(folder, new_folder, table_updates,
                  target=target, max_jobs=max_jobs)

    make_bdv_server_file([os.path.join(new_folder, 'images'),
                          os.path.join(new_folder, 'segmentations')],
                         os.path.join(new_folder, 'misc', 'bdv_server.txt'),
                         relative_paths=True)
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# TODO expose target and max_jobs as well
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update patch version of platy-browser-data.')
    parser.add_argument('--segmentation_names', type=str, nargs='+', default=[],
                        help="Names of the segmentations to update.")
    table_help_str = ("Names of the tables to update."
                      "The tables for segmentations in 'segmentation_names' will be updated without being passed here.")
    parser.add_argument('--table_names', type=str, nargs='+', default=[],
                        help=table_help_str)

    parser.add_argument('--description', type=str, default='',
                        help="Optional description for release message")
    parser.add_argument('--force_update', type=str2bool, default='no',
                        help="Create new release even if nothing needs to be updated.")

    args = parser.parse_args()
    update_patch(args.segmentation_names, args.table_names,
                 args.description, args.force_update)
