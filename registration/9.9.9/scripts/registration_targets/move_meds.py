import os
import json
from glob import glob
from shutil import move

from scripts.files.xml_utils import copy_xml_with_newpath


def move_meds():
    """ Move meds from rawdata folder to 0.0.0 to adhere
    to the new storage scheme for registration.
    """
    src_folder = '../data/rawdata'
    dst_folder = '../data/0.0.0/images'
    to_copy = glob(os.path.join(src_folder, 'prospr-6dpf*'))
    for p in to_copy:
        name = os.path.split(p)[1]
        dst = os.path.join(dst_folder, name)
        print(p, "to", dst)
        move(p, dst)


def update_version(version):
    """ Update MEDS in version folder to so that xmls point to MEDs in
    0.0.0 instead of raw data.
    """
    root_folder = '../data'
    src_folder = '../data/0.0.0/images'

    folder = os.path.join(root_folder, version)
    im_folder = os.path.join(folder, 'images')
    files = glob(os.path.join(im_folder, 'prospr-6dpf*'))

    # iterate over files, make sure the h5 file exists in src folder
    # and update reference in xml
    for ff in files:
        name = os.path.split(ff)[1]
        fin = os.path.join(src_folder, name)
        h5path = os.path.splitext(name)[0] + '.h5'
        h5path = os.path.join('../../0.0.0/images', h5path)
        assert os.path.exists(fin), fin
        assert os.path.exists(ff), ff
        copy_xml_with_newpath(fin, ff, h5path)


def update_meds():
    """ Update MEDs in all registered versions
    """
    version_file = '../data/versions.json'
    with open(version_file) as f:
        versions = json.load(f)

    for v in versions:
        if v == '0.0.0':
            continue
        print(v)
        update_version(v)


def diff_list():
    """ Get diff of old and new MEDs.
    """
    old_meds = glob('../data/0.0.0/images/prospr-6dpf-1*')
    new_meds = glob('../data/rawdata/prospr/*')

    old_normalized = [os.path.split(old)[1] for old in old_meds]
    old_normalized = [old.split('-')[-1].lower().split('.')[0]
                      if 'segmented' in old else old.split('-')[-2].lower()
                      for old in old_normalized]
    old_normalized = set(old_normalized)

    new_normalized = [os.path.split(new)[1] for new in new_meds]
    new_normalized = [new.split('-')[0].lower() for new in new_normalized]
    new_normalized = set(new_normalized)

    print("MEDS present in old folder, but not in the new one:")
    print(list(old_normalized - new_normalized))

    print("MEDS present in new folder, but not in the old one:")
    print(list(new_normalized - old_normalized))


if __name__ == '__main__':
    # move_meds()
    # update_meds()
    diff_list()
