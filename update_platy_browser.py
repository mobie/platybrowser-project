import os
import argparse
from subprocess import check_output, call

from scripts.attributes import make_cell_tables, make_nucleus_tables
from scripts.export import export_segmentation
from scripts.files import copy_tables, copy_segmentation, copy_static_files


# paths for paintera projects
# in order to get the new segmentation, changes need to be committed,
# s.t. they are stored in these files!
PAINTERA_ROOT = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
# TODO do we need the data postfix ???
PROJECT_CELLS = 'volumes/paintera/proofread_cells/data'
PROJECT_NUCLEI = 'volumes/paintera/nuclei/data'

# name for cell and nucleus segmentations
NAME_CELLS = 'em-segmented-cells-labels'
NAME_NUCLEI = 'em-segmented-nuclei-labels'


def check_inputs(update_cell_segmentation,
                 update_nucleus_segmentation,
                 update_cell_tables,
                 update_nucleus_tables):
    inputs = (update_cell_segmentation, update_nucleus_segmentation,
              update_cell_tables, update_nucleus_tables)
    have_changes = any(inputs)
    if update_cell_segmentation:
        update_cell_tables = True
    if update_nucleus_segmentation:
        update_nucleus_tables = True
    return have_changes, update_cell_tables, update_nucleus_tables


def get_tags():
    tag = check_output(['git', 'describe', '--tags']).decode('utf-8').rstrip('\n')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def make_folder_structure(tag):
    new_folder = os.makedirs('data', tag)
    # make all sub-folders
    os.makedirs(os.path.join(new_folder, 'tables'))
    os.makedirs(os.path.join(new_folder, 'images'))
    os.makedirs(os.path.join(new_folder, 'segmentations'))
    os.makedirs(os.path.join(new_folder, 'misc'))
    return new_folder


# TODO
# need lut from new to old segmentation ids
# to auto translate custom attributes
# also support extraction from paintera commit
def export_segmentations(folder, new_folder,
                         update_cell_segmentation,
                         update_nucleus_segmentation):
    # update or copy cell segmentation
    if update_cell_segmentation:
        export_segmentation(PAINTERA_ROOT, PROJECT_CELLS,
                            folder, new_folder, NAME_CELLS,
                            resolution=[.025, .02, .02])
    else:
        copy_segmentation(folder, new_folder, NAME_CELLS)

    # update or copy nucleus segmentation
    if update_nucleus_segmentation:
        export_segmentation(PAINTERA_ROOT, PROJECT_NUCLEI,
                            folder, new_folder, NAME_NUCLEI,
                            resolution=[.1, .08, .08])
    else:
        copy_segmentation(folder, new_folder, NAME_NUCLEI)


def make_attributes(folder, new_folder,
                    update_cell_tables, update_nucleus_tables):
    # update or copy cell tables
    if update_cell_tables:
        make_cell_tables(new_folder, NAME_CELLS)
    else:
        copy_tables(folder, new_folder, NAME_CELLS)

    # update or copy nucleus tables
    if update_nucleus_tables:
        make_nucleus_tables(new_folder, NAME_NUCLEI)
    else:
        copy_tables(folder, new_folder, NAME_NUCLEI)


# TODO check for errors
def make_release(tag):
    call(['git', 'commit', '-m', 'Automatic platybrowser update'])
    call(['git', 'tag', tag])
    # TODO autopush ???
    # call(['git', 'push', 'origin', 'master', '--tags'])


# TODO catch all exceptions and re-roll if exception was caught
def update_platy_browser(update_cell_segmentation=False,
                         update_nucleus_segmentation=False,
                         update_cell_tables=False,
                         update_nucleus_tables=False):
    """ Generate new version of platy-browser derived data.

    Arguments:
        update_cell_segmentation: Update the cell segmentation volume.
        update_nucleus_segmentation: Update the nucleus segmentation volume.
        update_cell_tables: Update the cell tables. This needs to be specified if the cell
            segmentation is not update, but the tables should be updated.
        update_nucleus_tables: Update the nucleus tables. This needs to be specified if the nucleus
            segmentation is not updated, but the tables should be updated.
    """

    # check inputs
    have_changes, update_cell_tables, update_nucleus_tables = check_inputs(update_cell_segmentation,
                                                                           update_nucleus_segmentation,
                                                                           update_cell_tables,
                                                                           update_nucleus_tables)
    if not have_changes:
        print("Nothing needs to be update, skipping ")
        return

    # we always increase the release tag (in the last digit)
    # when making a new version of segmentation or attributes
    tag, new_tag = get_tags()

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = make_folder_structure(new_tag)

    # export new segmentation(s)
    export_segmentations(folder, new_folder,
                         update_cell_segmentation,
                         update_nucleus_segmentation)

    # generate new attribute tables (if necessary)
    make_attributes(folder, new_folder,
                    update_cell_tables,
                    update_nucleus_tables)

    # copy files that were not touched
    copy_static_files(folder, new_folder)

    # make new release
    make_release(new_tag)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def table_help_str(name):
    help_str = """Update the %s tables.
                  Only needs to be specified if the %s segmentation is not updated,
                  but the tables should be updated."""
    return help_str % (name, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update derived data for the platy browser')

    parser.add_argument('--update_cell_segmentation', type=str2bool,
                        default=False, help="Update the cell segmentation.")
    parser.add_argument('--update_nucleus_segmentation', type=str2bool,
                        default=False, help="Update the nucleus segmentation.")
    parser.add_argument('--update_cell_tables', type=str2bool,
                        default=False, help=table_help_str("cell"))
    parser.add_argument('--update_nucleus_tables', type=str2bool,
                        default=False, help=table_help_str("nucleus"))

    args = parser.parse_args()
    update_platy_browser(args.update_cell_segmentation,
                         args.update_nucleus_segmentation,
                         args.update_cell_tables,
                         args.update_nucleus_tables)
