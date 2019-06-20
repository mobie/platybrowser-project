import os
import glob
from .xml_utils import copy_xml_with_abspath


def copy_files_with_pattern(src_folder, dst_folder, pattern):
    files = glob.glob(os.path.join(src_folder, pattern))
    for ff in files:
        ext = os.path.splitext(ff)[1]
        if ext != '.xml':
            continue
        xml_in = ff
        xml_out = os.path.join(dst_folder,
                               os.path.split(ff)[1])
        copy_xml_with_abspath(xml_in, xml_out)


def copy_tables(src_folder, dst_folder, name):
    pass


def copy_segmentation(src_folder, dst_folder, name):
    pass


def copy_static_files(src_folder, dst_folder):
    pass
