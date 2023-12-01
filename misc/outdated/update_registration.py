#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import os
import json
import argparse
from concurrent import futures
from glob import glob

import imageio
import luigi
import numpy as np
import pandas as pd
from pybdv import make_bdv
from pybdv.metadata import get_data_path

from mmpb.files import copy_and_check_image_dict, copy_release_folder
from mmpb.files.xml_utils import write_simple_xml, write_s3_xml
from mmpb.release_helper import (add_version, get_modality_names,
                                 get_version, make_folder_structure)
from mmpb.extension.registration import ApplyRegistrationLocal, ApplyRegistrationSlurm
from mmpb.default_config import get_default_shebang
from mmpb.attributes.base_attributes import base_attributes
from mmpb.attributes.genes import (create_auxiliary_gene_file,
                                   gene_assignment_table,
                                   vc_assignment_table)
from mmpb.util import add_max_id

with open('misc/prospr_name_dict.json') as f:
    PROSPR_NAME_DICT = json.load(f)


def get_tags():
    tag = get_version('data')
    new_tag = tag.split('.')
    new_tag[-1] = str(int(new_tag[-1]) + 1)
    new_tag = '.'.join(new_tag)
    return tag, new_tag


def parse_prospr(modality, name):
    name = os.path.split(name)[1]
    name = os.path.splitext(name)[0]
    name = name.split('--')[0]
    name = PROSPR_NAME_DICT[name]
    return '-'.join([modality, name])


def copy_file(in_path, out_path, resolution=[.55, .55, .55]):
    chunks = (96,) * 3
    if os.path.exists(out_path + '.xml'):
        return
    print("Copy", in_path, "to", out_path)
    vol = np.asarray(imageio.volread(in_path + '-ch0.tif'))
    downscale_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    make_bdv(vol, out_path + '.n5', downscale_factors,
             unit='micrometer', resolution=resolution,
             chunks=chunks)


def copy_to_bdv(inputs, output_folder, image_dict):
    print("Copy tifs to bdv/n5 in", output_folder)
    outputs = [os.path.join(output_folder, os.path.split(inp)[1]) for inp in inputs]
    n_jobs = 48
    with futures.ProcessPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(copy_file, inp, outp) for inp, outp in zip(inputs, outputs)]
        [t.result() for t in tasks]

    root_out = os.path.split(output_folder)[0]
    version = os.path.split(os.path.split(root_out)[0])[1]
    for inp, xml_in in zip(inputs, outputs):
        name = os.path.split(inp)[1]
        storage = image_dict[name]['Storage']
        if 'remote' in storage:
            xml_out = os.path.join(root_out, storage['remote'])
            path_in_bucket = os.path.join(version, 'images', 'remote', name + '.n5')
            write_s3_xml(xml_in + '.xml', xml_out, path_in_bucket)


def registration_impl(inputs, outputs, transformation_file, output_folder,
                      tmp_folder, target, max_jobs, image_dict,
                      interpolation, dtype='unsigned char'):
    task = ApplyRegistrationSlurm if target == 'slurm' else ApplyRegistrationLocal

    # write path name files to json
    input_file = os.path.join(tmp_folder, 'input_files.json')
    inputs = [os.path.abspath(inpath) for inpath in inputs]
    with open(input_file, 'w') as f:
        json.dump(inputs, f)

    output_file = os.path.join(tmp_folder, 'output_files.json')
    outputs = [os.path.abspath(outpath) for outpath in outputs]
    with open(output_file, 'w') as f:
        json.dump(outputs, f)

    # update the task config
    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    shebang = get_default_shebang()
    global_config = task.default_global_config()
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    task_config = task.default_task_config()
    task_config.update({'mem_limit': 16, 'time_limit': 240, 'threads_per_job': 4,
                        'ResultImagePixelType': dtype})
    with open(os.path.join(config_dir, 'apply_registration.config'), 'w') as f:
        json.dump(task_config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
             input_path_file=input_file, output_path_file=output_file,
             transformation_file=transformation_file, output_format='tif',
             interpolation=interpolation)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Registration failed")

    copy_to_bdv(outputs, output_folder, image_dict)


def apply_registration(input_folder, new_folder,
                       transformation_file, modality,
                       target, max_jobs, name_parser,
                       image_dict, interpolation='nearest'):
    tmp_folder = './tmp_registration'
    os.makedirs(tmp_folder, exist_ok=True)

    # find all input files (only support tif input)
    inputs = glob(os.path.join(input_folder, '*.tif'))

    if len(inputs) == 0:
        raise RuntimeError("Did not find any files for modality %s in %s" % (modality,
                                                                             input_folder))

    # we write temporary tifs with the fiji-elastix wrapper and then copy to bdv/n5
    output_folder = os.path.join(tmp_folder, 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_names = [name_parser(modality, name) for name in inputs]
    outputs = [os.path.join(output_folder, name) for name in output_names]

    output_folder = os.path.join(new_folder, 'images', 'local')
    registration_impl(inputs, outputs, transformation_file, output_folder,
                      tmp_folder, target, max_jobs, image_dict, interpolation=interpolation)


def remove_link(path):
    if os.path.exists(path) and os.path.islink(path):
        print("Remove link to previous table:", path)
        os.unlink(path)


def update_prospr(new_folder, input_folder, transformation_file,
                  target, max_jobs, image_dict):

    # update the auxiliaty gene volume
    image_folder = os.path.join(new_folder, 'images', 'local')
    aux_out_path = os.path.join(new_folder, 'misc', 'prospr-6dpf-1-whole_meds_all_genes.h5')
    if not os.path.exists(aux_out_path):
        create_auxiliary_gene_file(image_folder, aux_out_path)
        # write the new xml
        h5_path = os.path.split(aux_out_path)[1]
        xml_path = os.path.splitext(aux_out_path)[0] + '.xml'
        write_simple_xml(xml_path, h5_path, path_type='relative')

    # update the gene table
    seg_path = os.path.join(image_folder, 'sbem-6dpf-1-whole-segmented-cells.xml')
    seg_path = get_data_path(seg_path, return_absolute_path=True)
    assert os.path.exists(seg_path), seg_path

    table_folder = os.path.join(new_folder, 'tables', 'sbem-6dpf-1-whole-segmented-cells')
    default_table_path = os.path.join(table_folder, 'default.csv')
    table = pd.read_csv(default_table_path, sep='\t')
    labels = table['label_id'].values.astype('uint64')

    tmp_folder = './tmp_registration'
    gene_out_path = os.path.join(table_folder, 'genes.csv')
    # # we need to remove the link to the old gene table, if it exists
    remove_link(gene_out_path)
    gene_assignment_table(seg_path, aux_out_path, gene_out_path,
                          labels, tmp_folder, target)

    # register virtual cells
    vc_name = 'prospr-6dpf-1-whole-virtual-cells'
    vc_path = os.path.join(input_folder, 'virtual_cells', 'virtual_cells--prosprspaceMEDs.tif')
    inputs = [vc_path]
    outputs = [os.path.join(tmp_folder, 'outputs', vc_name)]
    tmp_folder_vc = os.path.join(tmp_folder, 'vc')
    os.makedirs(tmp_folder_vc, exist_ok=True)
    registration_impl(inputs, outputs, transformation_file, image_folder,
                      tmp_folder_vc, target, max_jobs, image_dict,
                      interpolation='nearest', dtype='unsigned short')

    # compute the table for the virtual cells
    vc_table_folder = os.path.join(new_folder, 'tables', vc_name)
    os.makedirs(vc_table_folder, exist_ok=True)
    vc_table = os.path.join(vc_table_folder, 'default.csv')
    remove_link(vc_table)
    vc_path = os.path.join(image_folder, vc_name + '.n5')
    key = 'setup0/timepoint0/s0'
    add_max_id(vc_path, key)

    assert os.path.exists(vc_path), vc_path
    resolution = [.55, .55, .55]
    base_attributes(vc_path, key, vc_table, resolution,
                    tmp_folder_vc, target, max_jobs,
                    correct_anchors=False)

    # update the vc based gene assignments as well
    vc_expression_path = os.path.join(new_folder, 'tables',
                                      'prospr-6dpf-1-whole-virtual-cells',
                                      'profile_clust_curated.csv')
    med_expression_path = gene_out_path
    vc_assignment_out = os.path.join(table_folder, 'vc_assignments.csv')
    remove_link(vc_assignment_out)
    vc_assignment_table(seg_path, vc_path, key,
                        vc_expression_path, med_expression_path,
                        vc_assignment_out, tmp_folder_vc, target)


# we should encode the source prefix and the transformation file to be used
# in a config file in the transformation folder
def update_registration(transformation_file, input_folder, modality,
                        target, max_jobs):
    """ Update the prospr registration.

    This is a special case of 'update_patch', that applies a new prospr registration.

    Arguments:
        transformation_file [str] - path to the transformation used to register
        input_folder [str] - folder with unregistered data
        modality [str] - data modality to apply the registration to
        target [str] - target of computation
        max_jobs [int] - max number of jobs for computation
    """
    tag, new_tag = get_tags()
    modality_names = get_modality_names('data', tag)
    if modality not in modality_names:
        raise ValueError("Invalid modality %s" % modality)

    print("Updating platy browser from", tag, "to", new_tag)

    # make new folder structure
    folder = os.path.join('data', tag)
    new_folder = os.path.join('data', new_tag)
    make_folder_structure(new_folder)

    # copy the release folder
    copy_release_folder(folder, new_folder, exclude_prefixes=[modality])

    if modality == "prospr-6dpf-1-whole":
        name_parser = parse_prospr
    else:
        raise NotImplementedError

    im_dict_path = os.path.join(folder, 'images', 'images.json')
    with open(im_dict_path) as f:
        image_dict = json.load(f)

    # apply new registration to all files of the source prefix
    transformation_file = os.path.abspath(transformation_file)
    apply_registration(input_folder, new_folder,
                       transformation_file, modality,
                       target, max_jobs, name_parser,
                       image_dict)

    if modality == "prospr-6dpf-1-whole":
        update_prospr(new_folder, input_folder, transformation_file,
                      target, max_jobs, image_dict)
    else:
        raise NotImplementedError

    # copy image dict and check that all image and table files are there
    copy_and_check_image_dict(folder, new_folder)

    add_version(new_tag, 'data')
    print("Updated platybrowser to new release", new_tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update registration to EM.')
    parser.add_argument('transformation_file', type=str, help="path to transformation file")

    parser.add_argument('--input_folder', type=str, default="data/rawdata/prospr",
                        help="Folder with (not registered) input files")

    modstr = "Input modality. Change this if not using 'input_folder' default value."
    parser.add_argument('--modality', type=str, default="prospr-6dpf-1-whole",
                        help=modstr)

    parser.add_argument('--target', type=str, default='slurm',
                        help="Computatin plaform, can be 'slurm' or 'local'")
    parser.add_argument('--max_jobs', type=int, default=100,
                        help="Maximal number of jobs used for computation")

    args = parser.parse_args()

    update_registration(args.transformation_file, args.input_folder, args.modality,
                        args.target, args.max_jobs)
