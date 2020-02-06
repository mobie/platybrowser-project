import os
import json
import shutil
from glob import glob
from mmpb.files.name_lookup import (look_up_filename, get_image_properties,
                                    DYNAMIC_SEGMENTATIONS, get_dynamic_segmentation_properties)
from mmpb.files.xml_utils import get_h5_path_from_xml, copy_xml_with_newpath

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
DRY_RUN = True


def new_folder_structure(folder):
    image_folder = os.path.join(folder, 'images')
    local_folder = os.path.join(image_folder, 'local')
    remote_folder = os.path.join(image_folder, 'remote')
    if DRY_RUN:
        print("Creating folder", local_folder)
        print("Creating folder", remote_folder)
    else:
        assert os.path.exists(image_folder), image_folder
        os.makedirs(local_folder, exist_ok=True)
        os.makedirs(remote_folder, exist_ok=True)


def move_image_file(image_folder, xml_path):
    name = os.path.splitext(os.path.split(xml_path)[1])[0]
    new_name = look_up_filename(name)

    # get the linked hdf5 path
    image_path = get_h5_path_from_xml(xml_path, return_absolute_path=True)

    # move the xml to 'images/local'
    new_xml_path = os.path.join(image_folder, 'local', new_name + '.xml')
    if DRY_RUN:
        print("Moving", xml_path, "to", new_xml_path)
    else:
        shutil.move(xml_path, new_xml_path)

    # if the hdf5 file is in the same folder, move it to 'images/local' as well
    h5_is_local = len(os.path.relpath(image_path, os.path.split(xml_path)[0]).split('/')) == 1
    if h5_is_local:
        new_image_path = os.path.join(image_folder, 'local', new_name + '.h5')
        if DRY_RUN:
            print("Moving", image_path, "to", new_image_path)
        else:
            assert os.path.exists(image_path), image_path
            shutil.move(image_path, new_image_path)
    # if not, construct the new correct data path
    else:
        # the new image path might be in rawdata; in this case there is now '/local'
        # subfolder, if it is in a version folder, it is in '/local'
        im_root, im_name = os.path.split(image_path)
        # take care of 'segmentations'
        if os.path.split(im_root)[1] == 'segmentations':
            im_root = os.path.join(os.path.split(im_root)[0], 'images')
        new_image_path = os.path.join(im_root, new_name + '.h5')
        if not os.path.exists(new_image_path):
            new_image_path = os.path.join(im_root, 'local', new_name + '.h5')

    new_rel_data_path = os.path.relpath(new_image_path, os.path.split(new_xml_path)[0])
    if DRY_RUN:
        print("Setting new xml path to", new_rel_data_path)
    else:
        assert os.path.exists(new_image_path), new_image_path
        # set path in xml
        copy_xml_with_newpath(new_xml_path, new_xml_path, new_rel_data_path)

    return new_name


def update_image_dict(image_folder, image_dict):
    dict_out_file = os.path.join(image_folder, 'images.json')
    if os.path.exists(dict_out_file):
        with open(dict_out_file) as f:
            image_dict.update(json.load(f))

    with open(dict_out_file, 'w') as f:
        json.dump(image_dict, f, sort_keys=True, indent=2)


def update_image_data(folder):
    image_dict = {}
    image_folder = os.path.join(folder, 'images')
    xmls = glob(os.path.join(image_folder, "*.xml"))

    for xml in xmls:
        name = move_image_file(image_folder, xml)
        image_dict.update({name: get_image_properties(name)})

    if DRY_RUN:
        print("New image dict:")
        # print(image_dict)
    else:
        update_image_dict(image_folder, image_dict)


# rename the table folders correctly
# fix links to account for the updated names
def update_tables(folder):
    table_folder = os.path.join(folder, 'tables')
    tables = os.listdir(table_folder)
    for name in tables:
        new_name = look_up_filename(name)
        path, new_path = os.path.join(table_folder, name), os.path.join(table_folder, new_name)
        if DRY_RUN:
            print("Rename", path, "to", new_path)
        else:
            os.rename(path, new_path)

        # update links if necessary
        table_files = glob(os.path.join(new_path, '*.csv'))
        for table_file in table_files:
            if os.path.isfile(table_file):
                continue

            # read the link location
            link_location = os.path.realpath(table_file)

            # check if this soft-link is still valid
            if os.path.exists(link_location):
                continue

            # otherwise try to link to the renamed table file
            link_folder, table_name = os.path.split(link_location)
            # some table names have this typo
            if table_name == 'base.csv':
                table_name = 'default.csv'
            link_folder = os.path.split(link_folder)[0]
            link_location = os.path.join(link_folder, new_name, table_name)
            assert os.path.exists(link_location), link_location

            if DRY_RUN:
                print("Moving link from", table_file, link_location)
            else:
                os.unlink(table_file)
                rel_link = os.path.relpath(link_location, os.path.split(table_file)[0])
                os.symlink(rel_link, table_file)


def update_segmentation_data(folder):
    image_dict = {}
    dynamic_seg_dict = {}

    image_folder = os.path.join(folder, 'images')
    seg_folder = os.path.join(folder, 'segmentations')
    xmls = glob(os.path.join(seg_folder, "*.xml"))

    for xml in xmls:
        name = move_image_file(image_folder, xml)
        image_dict.update({name: get_image_properties(name)})
        if name in DYNAMIC_SEGMENTATIONS:
            dynamic_seg_dict.update({name: get_dynamic_segmentation_properties(name)})

    if DRY_RUN:
        print("New image dict:")
        # print(image_dict)
    else:
        update_image_dict(image_folder, image_dict)

    if DRY_RUN:
        print("New dynamic seg dict")
        # print(dynamic_seg_dict)
    else:
        dynamic_seg_path = os.path.join(folder, 'misc', 'dynamic_segmentations.json')
        with open(dynamic_seg_path, 'w') as f:
            json.dump(dynamic_seg_dict, f, sort_keys=True, indent=2)


def clean_up(version_folder):
    # remove segmentation folder (needs to be empty!)
    seg_folder = os.path.join(version_folder, 'segmentations')
    if DRY_RUN:
        print("Removing", seg_folder)
    else:
        os.rmdir(seg_folder)

    # remove bdv server config
    bdv_server_config = os.path.join(version_folder, 'misc', 'bdv_server.txt')
    if DRY_RUN:
        print("Removing", bdv_server_config)
    else:
        if os.path.exists(bdv_server_config):
            os.remove(bdv_server_config)


# migrate version folder from old to new data layout
def migrate_version(version):
    version_folder = os.path.join(ROOT, version)

    # 1.) make new folder structure
    new_folder_structure(version_folder)

    # 2.) iterate over all images and segmentations, replace names (if necessary),
    # move the files and make the new images.json dict
    update_image_data(version_folder)

    # 3.) iterate over all table links and repair them
    update_segmentation_data(version_folder)

    # 4.) update the tables
    update_tables(version_folder)

    # 5.) clean up:
    clean_up(version_folder)


# migrate all the data in the raw folder
def migrate_rawfolder():
    raw_folder = os.path.join(ROOT, 'rawdata')
    xmls = glob(os.path.join(raw_folder, "*.xml"))

    for xml_path in xmls:
        name = os.path.splitext(os.path.split(xml_path)[1])[0]
        new_name = look_up_filename(name)

        # get the linked hdf5 path
        image_path = get_h5_path_from_xml(xml_path, return_absolute_path=True)

        # move the xml to 'images/local'
        new_xml_path = os.path.join(raw_folder, new_name + '.xml')
        if DRY_RUN:
            print("Moving", xml_path, "to", new_xml_path)
        else:
            shutil.move(xml_path, new_xml_path)

        new_image_path = os.path.join(raw_folder, new_name + '.h5')
        if DRY_RUN:
            print("Moving", image_path, "to", new_image_path)
        else:
            assert os.path.exists(image_path), image_path
            shutil.move(image_path, new_image_path)

        new_rel_data_path = new_name + '.h5'
        if DRY_RUN:
            print("Setting new xml path to", new_rel_data_path)
        else:
            assert os.path.exists(new_image_path), new_image_path
            # set path in xml
            copy_xml_with_newpath(new_xml_path, new_xml_path, new_rel_data_path)

        # rename the tables folder if it exists
        table_folder = os.path.join(raw_folder, 'tables', name)
        if os.path.exists(table_folder):
            new_table_folder = os.path.join(raw_folder, 'tables', new_name)
            if DRY_RUN:
                print("Rename", table_folder, "to", new_table_folder)
            else:
                os.rename(table_folder, new_table_folder)


# iterate over all the xmls in this version, follow the links
# and replace h5 files with n5 (if necessary)
def to_n5(version):
    pass


def make_remote_xmls(version):
    pass


def remove_deprecated_data():
    # cats-neuropil
    # traces
    # AcTub (this was removed at some point)
    # Pty2 (this was removed at some point)
    # meds_all_genes (does not belong in image folder)
    # xray (is not part of any version yet, but we need to move the raw data)

    def remove_deprecated_seg(folder, pattern):
        # remove xml for traces
        files = glob(os.path.join(vfolder, 'segmentations', pattern))
        if len(files) > 0:
            assert len(files) == 1
            if DRY_RUN:
                print("Remove", files[0])
            else:
                os.remove(files[0])

        # remove tables for traces
        files = glob(os.path.join(vfolder, 'tables', pattern))
        if len(files) > 0:
            assert len(files) == 1
            if DRY_RUN:
                print("Remove", files[0])
            else:
                shutil.rmtree(files[0])

    def remove_deprecated_im(folder, pattern):
        # remove xml for traces
        files = glob(os.path.join(vfolder, 'images', pattern))
        if len(files) > 0:
            assert len(files) == 1, str(files)
            if DRY_RUN:
                print("Remove", files[0])
            else:
                os.remove(files[0])

    # remove xmls from the version folders
    # (data from rawfolder should be backed up by hand!)
    version_folders = glob(os.path.join(ROOT, "0.*"))
    for vfolder in version_folders:
        remove_deprecated_seg(vfolder, '*traces*')
        remove_deprecated_seg(vfolder, '*cats*')
        remove_deprecated_im(vfolder, '*AcTub*')
        remove_deprecated_im(vfolder, '*Pty2*')
        remove_deprecated_im(vfolder, '*meds_all_genes*')


if __name__ == '__main__':
    # remove the data we don't want to upload (yet)
    # remove_deprecated_data()

    # change names and xmls in the rawfolder
    # migrate_rawfolder()

    version = '0.0.1'
    migrate_version(version)
