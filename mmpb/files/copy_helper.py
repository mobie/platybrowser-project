import os
import shutil
import numpy as np

from elf.io import open_file
from pybdv.converter import copy_dataset
from pybdv.metadata import write_n5_metadata
from pybdv.util import get_key, get_number_of_scales, get_scale_factors

from .xml_utils import copy_xml_with_newpath, get_h5_path_from_xml
from .sources import get_image_names, get_segmentation_names, get_segmentations
from ..attributes.base_attributes import write_additional_table_file


def make_squashed_link(src_file, dst_file, override=False):

    if os.path.exists(dst_file):
        if override and os.path.islink(dst_file):
            os.unlink(dst_file)
        elif override and not os.path.islink(dst_file):
            raise RuntimeError("Cannot override an actual file!")
        elif not override:
            return

    if os.path.islink(src_file):
        src_file = os.path.realpath(src_file)
    dst_folder = os.path.split(dst_file)[0]
    rel_path = os.path.relpath(src_file, dst_folder)
    os.symlink(rel_path, dst_file)


def copy_file(xml_in, xml_out):
    h5path = get_h5_path_from_xml(xml_in, return_absolute_path=True)
    xml_dir = os.path.split(xml_out)[0]
    h5path = os.path.relpath(h5path, start=xml_dir)
    copy_xml_with_newpath(xml_in, xml_out, h5path, path_type='relative')


def copy_tables(src_folder, dst_folder, name):
    table_in = os.path.join(src_folder, 'tables', name)
    table_out = os.path.join(dst_folder, 'tables', name)
    os.makedirs(table_out, exist_ok=True)

    table_files = os.listdir(table_in)
    table_files = [ff for ff in table_files if os.path.splitext(ff)[1] == '.csv']

    for ff in table_files:
        src_file = os.path.join(table_in, ff)
        dst_file = os.path.join(table_out, ff)

        make_squashed_link(src_file, dst_file)

    # write the txt file for additional tables
    write_additional_table_file(table_out)


def copy_segmentation(src_folder, dst_folder, name):
    # copy the segmentation xml
    name_with_ext = '%s.xml' % name
    xml_in = os.path.join(src_folder, 'segmentations', name_with_ext)
    if not os.path.exists(xml_in):
        raise RuntimeError("Could not find %s in the src folder %s" % (name, src_folder))
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


def copy_image_data(src_folder, dst_folder, exclude_prefixes=[]):
    # get all image names that need to be copied
    names = get_image_names()

    for name in names:

        prefix = name.split('-')[:4]
        prefix = '-'.join(prefix)

        if prefix in exclude_prefixes:
            continue

        name += '.xml'
        in_path = os.path.join(src_folder, 'images', name)
        out_path = os.path.join(dst_folder, 'images', name)
        if not os.path.exists(in_path):
            raise RuntimeError("Could not find %s in the src folder %s" % (name, src_folder))
        # copy the xml
        copy_file(in_path, out_path)


def copy_misc_data(src_folder, dst_folder):
    # copy the aux gene data
    prospr_prefix = 'prospr-6dpf-1-whole'
    aux_name = '%s_meds_all_genes.xml' % prospr_prefix
    copy_file(os.path.join(src_folder, 'misc', aux_name),
              os.path.join(dst_folder, 'misc', aux_name))

    # copy the bookmarks
    bkmrk_in = os.path.join(src_folder, 'misc', 'bookmarks.json')
    if os.path.exists(bkmrk_in):
        shutil.copyfile(bkmrk_in,
                        os.path.join(dst_folder, 'misc', 'bookmarks.json'))


def copy_segmentations(src_folder, dst_folder, exclude_prefixes=[]):
    names = get_segmentation_names()
    for name in names:

        prefix = name.split('-')[:4]
        prefix = '-'.join(prefix)

        if prefix in exclude_prefixes:
            continue

        copy_segmentation(src_folder, dst_folder, name)


def copy_all_tables(src_folder, dst_folder):
    segmentations = get_segmentations()
    for name, seg in segmentations.items():
        has_table = seg.get('has_tables', False) or 'table_update_function' in seg
        if not has_table:
            continue
        copy_tables(src_folder, dst_folder, name)


def copy_release_folder(src_folder, dst_folder, exclude_prefixes=[]):
    # copy static image and misc data
    copy_image_data(src_folder, dst_folder, exclude_prefixes)
    copy_misc_data(src_folder, dst_folder)
    copy_segmentations(src_folder, dst_folder, exclude_prefixes)
    copy_all_tables(src_folder, dst_folder)


def normalize_scale_factors(scale_factors, start_scale):
    # we expect scale_factors[0] == [1 1 1]
    assert np.prod(scale_factors[0]) == 1

    # convert to relative scale factors
    rel_scales = [scale_factors[0]]
    for scale in range(1, len(scale_factors)):
        rel_factor = [sf / prev_sf for sf, prev_sf in zip(scale_factors[scale],
                                                          scale_factors[scale - 1])]
        rel_scales.append(rel_factor)

    # return the relative scales starting at the new scale
    new_factors = [[1., 1., 1.]] + rel_scales[(start_scale + 1):]
    return new_factors


def copy_to_bdv_n5(in_file, out_file, chunks, resolution,
                   n_threads=32, start_scale=0):

    n_scales = get_number_of_scales(in_file, 0, 0)
    scale_factors = get_scale_factors(in_file, 0)
    # double check newly implemented functions in pybdv
    assert n_scales == len(scale_factors)

    scale_factors = normalize_scale_factors(scale_factors, start_scale)

    for out_scale, in_scale in enumerate(range(start_scale, n_scales)):
        in_key = get_key(True, 0, 0, in_scale)
        out_key = get_key(False, 0, 0, out_scale)

        if chunks is None:
            with open_file(in_file, 'r') as f:
                chunks_ = f[in_key].chunks
        else:
            chunks_ = chunks

        copy_dataset(in_file, in_key, out_file, out_key, False,
                     chunks_, n_threads)

    write_n5_metadata(out_file, scale_factors, resolution, setup_id=0)
