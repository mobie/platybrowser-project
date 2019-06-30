import os
import glob
import shutil
from .xml_utils import copy_xml_with_newpath, get_h5_path_from_xml


def copy_file(xml_in, xml_out):
    h5path = get_h5_path_from_xml(xml_in, return_absolute_path=True)
    xml_dir = os.path.split(xml_out)[0]
    h5path = os.path.relpath(h5path, start=xml_dir)
    copy_xml_with_newpath(xml_in, xml_out, h5path, path_type='relative')


def copy_files_with_pattern(src_folder, dst_folder, pattern):
    files = glob.glob(os.path.join(src_folder, pattern))
    for ff in files:
        ext = os.path.splitext(ff)[1]
        if ext != '.xml':
            continue
        xml_in = ff
        xml_out = os.path.join(dst_folder, os.path.split(ff)[1])
        copy_file(xml_in, xml_out)


def copy_tables(src_folder, dst_folder, name):
    pass


def copy_segmentation(src_folder, dst_folder, name):
    name_with_ext = '%s.xml' % name
    xml_in = os.path.join(src_folder, 'segmentations', name_with_ext)
    xml_out = os.path.join(dst_folder, 'segmentations', name_with_ext)
    copy_file(xml_in, xml_out)


def copy_image_data(src_folder, dst_folder):
    # copy sbem image data
    sbem_prefix = 'sbem-6dpf-1-whole'
    raw_name = '%s-raw.xml' % sbem_prefix
    copy_file(os.path.join(src_folder, raw_name),
              os.path.join(dst_folder, raw_name))

    # copy the prospr med image data
    copy_files_with_pattern(src_folder, dst_folder, '*-MED*')

    # copy the segmented prospr regions
    copy_files_with_pattern(src_folder, dst_folder, 'prospr-6dpf-1-whole-segmented-*')


def copy_misc_data(src_folder, dst_folder):
    # copy the aux gene data
    prospr_prefix = 'prospr-6dpf-1-whole'
    aux_name = '%s_meds_all_genes.xml' % prospr_prefix
    copy_file(os.path.join(src_folder, aux_name),
              os.path.join(dst_folder, aux_name))

    # copy the bookmarks
    bkmrk_in = os.path.join(src_folder, 'bookmarks.json')
    if os.path.exists(bkmrk_in):
        shutil.copyfile(bkmrk_in,
                        os.path.join(dst_folder, 'bookmarks.json'))


# TODO
def copy_static_segmentations(src_folder, dst_folder):
    pass
