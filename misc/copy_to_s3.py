#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
from subprocess import run
from glob import glob
import s3fs
from mmpb.files.xml_utils import read_path_in_bucket

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
# don't copy raw-data and cell segmentation for now
EXCLUDE = ["/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.5/images/local/sbem-6dpf-1-whole-segmented-cells.n5"]


def copy_n5_to_s3(path, path_in_bucket):
    full_s3_path = os.path.join('embl', 'platybrowser', path_in_bucket)
    # mc copy is very picky about putting '/' at the end.
    # only the following combination works properly:
    # 'src/', 'target/'
    cmd = ['mc', 'cp', '-r', path + '/', full_s3_path + '/']
    print("Run command", cmd)
    run(cmd)


def copy_all_to_s3():
    copy_in = '/g/kreshuk/pape/copied_to_n5.json'
    with open(copy_in) as f:
        copied = json.load(f)

    copy_out = '/g/kreshuk/pape/copied_to_s3.json'
    if os.path.exists(copy_out):
        with open(copy_out) as f:
            s3_copied = json.load(f)
    else:
        s3_copied = []

    for ff in copied:
        data_path = os.path.splitext(ff)[0] + '.n5'
        if data_path in s3_copied:
            continue
        if data_path in EXCLUDE:
            continue

        if not os.path.exists(data_path):
            print("Expected path", data_path, "does not exist!")
            break

        path_in_bucket = os.path.relpath(data_path, ROOT)
        if 'local' in path_in_bucket:
            path_in_bucket = path_in_bucket.replace('local', 'remote')
        copy_n5_to_s3(data_path, path_in_bucket)
        s3_copied.append(data_path)

    with open(copy_out, 'w') as f:
        json.dump(s3_copied, f)


def check_xml(fs, xml, check_bdv=False):
    bucket_name = 'platybrowser'
    path_in_bucket = read_path_in_bucket(xml)
    full_path = os.path.join(bucket_name, path_in_bucket)
    have_file = fs.exists(full_path)
    if check_bdv:
        have_bdv = fs.exists(os.path.join(full_path, 'setup0'))
        return have_file and have_bdv
    else:
        return have_file


def check_all_xml():
    server = 'https://s3.embl.de'
    fs = s3fs.S3FileSystem(anon=True,
                           client_kwargs={'endpoint_url': server})

    folder = os.path.join(ROOT, '0.6.6', 'images', 'remote')
    xmls = glob(os.path.join(folder, '*.xml'))
    n_missing = 0
    for xml in xmls:
        have_file = check_xml(fs, xml, True)
        if not have_file:
            print("S3 file is missing for", xml)
            n_missing += 1
    print("Checked all xmls found", n_missing, "missing s3 files of", len(xmls))


if __name__ == '__main__':
    copy_all_to_s3()
    # check_all_xml()
