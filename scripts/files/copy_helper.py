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


# For now we put symlinks with relative paths, but I am not sure
# if this is the best idea, because I don't know if it will work on windows
def copy_tables(src_folder, dst_folder, name):
    table_in = os.path.join(src_folder, 'tables', name)
    table_out = os.path.join(dst_folder, 'tables', name)
    os.makedirs(table_out, exist_ok=True)

    table_files = os.listdir(table_in)
    table_files = [ff for ff in table_files if os.path.splitext(ff)[1] == '.csv']

    for ff in table_files:
        src_file = os.path.join(table_in, ff)
        dst_file = os.path.join(table_out, ff)

        rel_path = os.path.relpath(src_file, table_out)
        if not os.path.exists(dst_file):
            os.symlink(rel_path, dst_file)


def copy_segmentation(src_folder, dst_folder, name):
    # copy the segmentation xml
    name_with_ext = '%s.xml' % name
    xml_in = os.path.join(src_folder, 'segmentations', name_with_ext)
    xml_out = os.path.join(dst_folder, 'segmentations', name_with_ext)
    copy_file(xml_in, xml_out)

    # make link to the previous id look-up-table (if present)
    lut_name = 'new_id_lut_%s.json' % name
    lut_in = os.path.join(src_folder, 'misc', lut_name)
    if not os.path.exists(lut_in):
        return
    lut_out = os.path.join(dst_folder, 'misc', lut_name)
    if not os.path.exists(lut_out):
        rel_path = os.path.relpath(lut_in, os.path.split(lut_out)[0])
        os.symlink(rel_path, lut_out)


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
