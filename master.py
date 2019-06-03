import os
import argparse
from subprocess import check_output, call

from scripts.attributes import base_attributes
from scripts.export import to_bdv


def get_tags():
    # TODO check for errors
    tag = check_output(['git', 'describe', '--tags']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


# TODO do we need more sub-folders ?
def make_folder_structure(tag):
    new_folder = os.makedirs('data', tag)
    new_tables_folder = os.path.join(new_folder, 'tables')
    os.makedirs(new_tables_folder)
    return new_folder


# TODO
# need lut from new to old segmentation ids
# to auto translate custom attributes
# also support extraction from paintera commit
def export_segmentation(input_path, input_key, output_path):
    resolution = [.025, .02, .02]
    to_bdv(input_path, input_key, output_path, resolution)


# TODO additional attributes
# - nucleus cell mapping
# - valentyna's attributes
# - kimberly's attributes
def make_attributes(input_path, input_key, output_path):
    base_attributes(input_path, output_key, output_path)


# TODO check for errors
def make_release(tag):
    call(['git', 'commit', '-m', 'Automatic platybrowser update'])
    call(['git', 'tag', tag])
    # TODO autopush ???
    # call(['git', 'push', 'origin', 'master', '--tags'])


# TODO catch all exceptions and re-roll if exception was caught
# TODO what arguments do we expose
# TODO need to deal with different kinds of segmentations
def master():
    """ Generate new version of platy-browser derived data.

    Arguments:
        update_cell_segmentation: Update the cell segmentation volume.
    """

    # we always increase the release tag (in the last digit)
    # when making a new version of segmentation or attributes
    tag, new_tag = get_tags()

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = make_folder_structure(new_tag)

    # TODO
    # Need to check if we actually need to export the new segmentation
    # export new segmentation(s)
    # export_segmentation()

    # TODO
    # generate new attribute tables (if necessary)
    # make_attributes()

    # TODO
    # ...

    # TODO
    # copy files that were not touched

    # make new release
    # make_release(new_tag)


if __name__ == '__main__':
    master()
