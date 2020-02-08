#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from copy import deepcopy
from subprocess import check_output

import mmpb.attributes
from mmpb.bookmarks import update_bookmarks
from mmpb.export import export_segmentation
from mmpb.files import (copy_and_check_image_dict, copy_image_data,
                        copy_misc_data, copy_segmentation, copy_tables)
from mmpb.release_helper import add_version, make_folder_structure
from mmpb.util import read_resolution


def get_tags():
    tag = check_output(['git', 'describe', '--abbrev=0']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def _load_dicts(folder):
    image_dict = os.path.join(folder, 'images', 'images.json')
    with open(image_dict) as f:
        image_dict = json.load(f)
    update_dict = os.path.join(folder, 'misc', 'dynamic_segmentations.json')
    with open(update_dict) as f:
        update_dict = json.load(f)
    return image_dict, update_dict


def update_segmentation(name, properties, update_config,
                        folder, new_folder, target, max_jobs):
    tmp_folder = 'tmp_export_%s' % name
    paintera_root, paintera_key = update_config['PainteraProject']
    pp_config = update_config.get('Postprocess', None)
    map_to_background = update_config.get('MapToBackground', None)

    # make the output path
    storage = properties['Storage']
    out_path = os.path.join(new_folder, 'images', storage['local'])

    # FIXME this is currently not working and needs to be adapted to new storage layout
    # (especially to n5 output data)
    export_segmentation(paintera_root, paintera_key, name,
                        folder, new_folder, out_path, tmp_folder,
                        pp_config=pp_config, map_to_background=map_to_background,
                        target=target, max_jobs=max_jobs)
    # TODO
    # make the s3 xml if we have remote storage
    if 'remote' in storage:
        pass


def update_segmentations(folder, new_folder, names_to_update, target, max_jobs):
    image_dict, update_dict = _load_dicts(folder)
    for name, properties in image_dict.items():
        # only update or copy for segmentations, which have
        # 'segmented' in their name
        if 'segmented' not in name:
            continue
        if name in names_to_update:
            update_segmentation(name, properties, update_dict[name],
                                folder, new_folder, target, max_jobs)
        else:
            copy_segmentation(folder, new_folder, name, properties)


def update_tables(folder, new_folder,
                  names_to_update, seg_update_names,
                  target, max_jobs):
    image_dict, update_dict = _load_dicts(folder)

    # first copy all tables that just need to be copied
    for name, properties in image_dict.items():
        table_folder = properties.get("TableFolder", None)
        needs_update = name in names_to_update
        if (table_folder is None) or needs_update:
            continue
        copy_tables(folder, new_folder, table_folder)

    # now update all tables that need to be updated
    for name in names_to_update:
        properties = image_dict[name]
        table_folder = properties.get("TableFolder", None)
        properties = update_dict[name]
        update_function = properties.get('TableUpdateFunction', None)
        if table_folder is None or update_function is None:
            raise RuntimeError("Tables for segmentation %s cannot be updated:" % name)

        tmp_folder = 'tmp_tables_%s' % name
        update_function = getattr(mmpb.attributes, update_function)
        paintera_path, paintera_key = properties['PainteraProject']
        resolution = read_resolution(paintera_path, paintera_key, to_um=True)
        seg_has_changed = name in seg_update_names
        update_function(folder, new_folder, name, tmp_folder, resolution,
                        target=target, max_jobs=max_jobs,
                        seg_has_changed=seg_has_changed)


def check_requested_updates(names_to_update, folder):
    image_dict, update_dict = _load_dicts(folder)
    for name in names_to_update:
        if name not in image_dict:
            raise ValueError("Requested update for %s, which does not exist" % name)
        if 'segmented' not in name:
            raise ValueError("Requested update for %s, which is not a segmentation" % name)
        if name not in update_dict:
            raise ValueError("Requested update for %s, which is a static segmentation" % name)


def update_patch(update_seg_names, update_table_names,
                 bookmarks=None, target='slurm', max_jobs=250):
    """ Generate new patch version of platy-browser derived data.

    The patch version is increased if derived data changes, e.g. by
    incorporating corrections for a segmentation or updating tables.

    Arguments:
        update_seg_names [list[str]] - names of segmentations to be updated.
        update_table_names [list[str]] - names of tables to be updated.
            Not that these only need to be specified if the corresponding segmentation is not
            updated, but the tables should be updated.
        bookmarks [dict] - bookmarks that will be added to this release (default: None)
        target [str] - target for the computation ('local' or 'slurm', default is 'slurm').
        max_jobs [int] - maximal number of jobs used for computation (default: 250).
    """

    # check if we have anything to update
    have_seg_updates = len(update_seg_names) > 0
    have_table_updates = len(update_table_names) > 0
    if not have_seg_updates and not have_table_updates:
        raise ValueError("No updates where provdied and force_update was not set")

    # increase the patch (last digit) release tag
    tag, new_tag = get_tags()
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)

    table_updates = deepcopy(update_seg_names)
    table_updates.extend(update_table_names)
    check_requested_updates(table_updates, folder)
    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    make_folder_structure(new_folder)

    # copy static image and misc data
    copy_image_data(folder, new_folder)
    copy_misc_data(folder, new_folder)

    # updated bookmarks if given
    if bookmarks is not None:
        update_bookmarks(new_folder, bookmarks)

    # export new segmentations
    update_segmentations(folder, new_folder, update_seg_names,
                         target=target, max_jobs=max_jobs)

    # generate new attribute tables
    update_tables(folder, new_folder, table_updates, update_seg_names,
                  target=target, max_jobs=max_jobs)

    # copy image dict and check that all image and table files are there
    copy_and_check_image_dict(folder, new_folder)

    add_version(new_tag)
    print("Updated platybrowser to new release", new_tag)
    # TODO
    # print instructions on how to make release, upload to embl.s3/platybrowser
    # and how to clean up


if __name__ == '__main__':
    help_str = """Path to a json containing list of the data to update.
                  See docstring of 'update_patch' for details."""
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

    bookmarks = update_dict.get('bookmarks', None)

    update_patch(update_dict['segmentations'], update_dict['tables'], bookmarks,
                 target=args.target, max_jobs=args.max_jobs)
