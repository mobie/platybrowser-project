import os
import xml.etree.ElementTree as ET
import numpy as np
from shutil import copyfile

from scripts.files.xml_utils import get_h5_path_from_xml
from glob import glob
from elf.io import open_file
from pybdv.converter import copy_dataset
from pybdv.util import get_key, get_number_of_scales, get_scale_factors
from pybdv.metadata import write_n5_metadata, get_resolution, indent_xml


def normalize_scale_factors(scale_factors, start_scale):
    if start_scale == 0:
        return scale_factors

    # we expect scale_factors[0] == [1 1 1]
    assert np.prod(scale_factors[0]) == 1

    # convert to relative scale factors
    rel_scales = [scale_factors[0]]
    for scale in range(1, len(scale_factors)):
        rel_factor = [sf / prev_sf for sf, prev_sf in zip(scale_factors[scale],
                                                          scale_factors[scale - 1])]
        rel_scales.append(rel_factor)

    # start at new scale
    new_factors = [[1, 1, 1]] + rel_scales[(start_scale + 1):]

    # back to absolute factors
    for scale in range(1, len(new_factors)):
        new_factor = [sf * prev_sf for sf, prev_sf in zip(new_factors[scale],
                                                          new_factors[scale - 1])]
        new_factors[scale] = new_factor

    return new_factors


def copy_file_to_bdv_n5(in_file, out_file, resolution,
                        chunks=None, start_scale=0):
    # if we have the out-file already, do nothing
    if os.path.exists(out_file):
        return

    n_threads = 16
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


# TODO move to pybdv
def make_xml_s3(in_file, out_file, path_in_bucket,
                s3_config, shape, resolution=None):
    nt = 1
    setup_id = 0
    setup_name = None

    setup_name = 'Setup%i' % setup_id if setup_name is None else setup_name
    nz, ny, nx = tuple(shape)

    # check if we have an xml already
    tree = ET.parse(in_file)
    root = tree.getroot()

    # load the sequence description
    seqdesc = root.find('SequenceDescription')

    # update the image loader
    # remove the old image loader
    imgload = seqdesc.find('ImageLoader')
    seqdesc.remove(imgload)

    # write the new image loader
    imgload = ET.SubElement(seqdesc, 'ImageLoader')
    bdv_dtype = 'bdv.n5.s3'
    imgload.set('format', bdv_dtype)
    el = ET.SubElement(imgload, 'Key')
    el.text = path_in_bucket

    # TODO read this from the s3 config instead
    el = ET.SubElement(imgload, 'ServiceEndpoint')
    el.text = 'https://s3.embl.de'
    el = ET.SubElement(imgload, 'BucketName')
    el.text = 'platybrowser'
    el = ET.SubElement(imgload, 'SigningRegion')
    el.text = 'us-west-2'

    # load the view descriptions
    viewsets = seqdesc.find('ViewSetups')
    # load the registration decriptions
    vregs = root.find('ViewRegistrations')

    # write new resolution and shape
    oz, oy, ox = 0.0, 0.0, 0.0
    dz, dy, dx = resolution
    vs = viewsets.find('ViewSetup')
    vss = vs.find('size')
    vss.text = '{} {} {}'.format(nx, ny, nz)
    vox = vs.find('voxelSize')
    voxs = vox.find('size')
    voxs.text = '{} {} {}'.format(dx, dy, dz)

    for t in range(nt):
        vreg = vregs.find('ViewRegistration')
        vt = vreg.find('ViewTransform')
        vt.set('type', 'affine')
        vta = vt.find('affine')
        vta.text = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox,
                                                                      dy, oy,
                                                                      dz, oz)
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(out_file)


def copy_images(in_folder, out_folder, data_out_folder,
                s3_config, images_to_copy, output_root):
    os.makedirs(out_folder, exist_ok=True)
    xml_s3_folder = os.path.join(out_folder, 's3-n5')
    os.makedirs(xml_s3_folder, exist_ok=True)
    # TODO make this one as well and copy xml ?
    # xml_h5_folder = os.path.join(out_folder, 'embl-h5')

    image_names = list(images_to_copy.keys())
    input_files = glob(os.path.join(in_folder, '*.xml'))
    input_names = [os.path.splitext(os.path.split(im)[1])[0]
                   for im in input_files]
    assert all(im in input_names for im in image_names), str(image_names)
    files_to_copy = [input_files[input_names.index(im)]
                     for im in image_names]

    for im_name, in_file in zip(image_names, files_to_copy):
        print("Copying", im_name, "...")
        in_h5 = get_h5_path_from_xml(in_file, True)
        # TODO we don't want to always copy to rawdata, but instead we
        # need to copy to the correct path extract from the old path
        out_file = os.path.join(data_out_folder, im_name + '.n5')

        options = images_to_copy[im_name]
        start_scale = options.get('start_scale', 0)
        chunks = options.get('chunks', None)
        resolution = options.get('resolution', None)
        # read the resolution from the xml if it is None
        if resolution is None:
            resolution = get_resolution(in_file)

        # copy from hdf5 to n5
        copy_file_to_bdv_n5(in_h5, out_file, resolution, chunks, start_scale)
        key = get_key(False, 0, 0, start_scale)
        with open_file(out_file, 'r') as f:
            shape = f[key].shape

        # update and copy the xml
        # path in bucket is the relative path from out_file to output_root
        path_in_bucket = os.path.relpath(out_file, output_root)
        out_file = os.path.join(xml_s3_folder, im_name + '.xml')
        make_xml_s3(in_file, out_file, path_in_bucket, s3_config, shape, resolution)


def copy_segmentations(in_folder, out_folder, segmentations_to_copy, output_root):
    # segmentation folders
    seg_in = os.path.join(in_folder, 'segmentations')
    seg_out = os.path.join(out_folder, 'segmentations')
    os.makedirs(seg_out, exist_ok=True)
    s3_folder = os.path.join(seg_out, 's3-n5')
    os.makedirs(s3_folder, exist_ok=True)
    # TODO make this one as well and copy xml ?
    # xml_h5_folder = os.path.join(out_folder, 'embl-h5')

    # table folders
    table_in = os.path.join(in_folder, 'tables')
    table_out = os.path.join(out_folder, 'tables')
    os.makedirs(table_out, exist_ok=True)

    seg_names = list(segmentations_to_copy.keys())
    input_files = glob(os.path.join(seg_in, '*.xml'))
    input_names = [os.path.splitext(os.path.split(im)[1])[0]
                   for im in input_files]
    assert all(im in input_names for im in seg_names), str(seg_names)
    files_to_copy = [input_files[input_names.index(im)]
                     for im in seg_names]

    for seg_name, in_file in zip(seg_names, files_to_copy):
        print("Copying", seg_name, "...")
        in_h5 = get_h5_path_from_xml(in_file, True)
        # TODO we don't want to always copy to rawdata, but instead we
        # need to copy to the correct path extract from the old path
        out_file = os.path.join(s3_folder, seg_name + '.n5')

        options = segmentations_to_copy[seg_name]
        start_scale = options.get('start_scale', 0)
        chunks = options.get('chunks', None)
        resolution = options.get('resolution', None)
        # read the resolution from the xml if it is None
        if resolution is None:
            resolution = get_resolution(in_file)

        # copy from hdf5 to n5
        copy_file_to_bdv_n5(in_h5, out_file, resolution, chunks, start_scale)
        key = get_key(False, 0, 0, start_scale)
        with open_file(out_file, 'r') as f:
            shape = f[key].shape

        # update and copy the xml
        # path in bucket is the relative path from out_file to output_root
        path_in_bucket = os.path.relpath(out_file, output_root)
        out_file = os.path.join(s3_folder, seg_name + '.xml')
        make_xml_s3(in_file, out_file, path_in_bucket, s3_config, shape, resolution)

        # check if we need to copy tables
        seg_table_in = os.path.join(table_in, seg_name)
        if not os.path.exists(seg_table_in):
            continue

        # TODO! we don't want to copy the tables, but just put relative symlinks.
        # only doing an explicit copy for the test!
        # copy all tables
        seg_table_out = os.path.join(table_out, seg_name)
        os.makedirs(seg_table_out, exist_ok=True)
        in_tables = glob(os.path.join(seg_table_in, '*'))
        for in_table in in_tables:
            table_name = os.path.split(in_table)[1]
            out_table = os.path.join(seg_table_out, table_name)
            if os.path.islink(in_table):
                in_table = os.path.abspath(os.path.realpath(in_table))
            copyfile(in_table, out_table)


# TODO allow changing chunks and lower start scale
def copy_folder_for_s3(version, images_to_copy, segmentations_to_copy, output_root, s3_config):
    input_root = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
    version_folder = os.path.join(input_root, version)
    assert os.path.exists(version_folder), version

    data_folder = os.path.join(output_root, 'rawdata')
    out_folder = os.path.join(output_root, version)

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # copy images:
    image_in = os.path.join(version_folder, 'images')
    image_out = os.path.join(out_folder, 'images')
    copy_images(image_in, image_out, data_folder,
                s3_config, images_to_copy, output_root)

    # copy segmentations and tables:
    copy_segmentations(version_folder, out_folder,
                       segmentations_to_copy, output_root)


if __name__ == '__main__':
    res = [.1, .08, .08]
    im_names = {'sbem-6dpf-1-whole-raw': {'start_scale': 3, 'resolution': res},
                'prospr-6dpf-1-whole-AChE-MED': {'resolution': [.55, .55, .55]}}
    seg_names = {'sbem-6dpf-1-whole-segmented-cells-labels': {'start_scale': 2,
                                                              'resolution': res,
                                                              'chunks': [32, 512, 512]}}
    out = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/test_n5'
    s3_config = {}
    copy_folder_for_s3('0.6.5', im_names, seg_names, out, s3_config)
