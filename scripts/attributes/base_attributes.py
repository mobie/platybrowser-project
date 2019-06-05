#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python
# TODO new platy-browser env

import os
import csv
import json
import z5py
import numpy as np

import luigi
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.morphology import CorrectAnchorsWorkflow


def make_config(tmp_folder):
    configs = MorphologyWorkflow.get_config()
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    global_config = configs['global']
    # TODO use new platy browser env
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_config['shebang'] = shebang
    global_config['block_shape'] = [64, 512, 512]
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)


def n5_attributes(input_path, input_key, tmp_folder, target, max_jobs):
    task = MorphologyWorkflow

    out_path = os.path.join(tmp_folder, 'data.n5')
    config_folder = os.path.join(tmp_folder, 'configs')

    out_key = 'attributes'
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs, target=target,
             config_dir=config_folder,
             input_path=input_path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix='attributes', max_jobs_merge=min(32, max_jobs))
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Attribute workflow failed")
    return out_path, out_key


# correct anchor points that are not inside of objects:
# for each object, check if the anchor point is in the object
# if it is NOT, do:
# - for all chunks that overlap with the bounding box of the object:
# - check if object is in chunk
# - if it is: set anchor to eccentricity center of object in chunk
# - else: continue
def run_correction(input_path, input_key, attr_path, attr_key,
                   tmp_folder, target, max_jobs):
    task = CorrectAnchorsWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=input_path, input_key=input_key,
             morphology_path=attr_path, morphology_key=attr_key)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Anchor correction failed")


def to_csv(input_path, input_key, output_path, resolution):
    # load the attributes from n5
    with z5py.File(input_path, 'r') as f:
        attributes = f[input_key][:]
    label_ids = attributes[:, 0:1]

    # the colomn names
    col_names = ['label_ids',
                 'anchor_x', 'anchor_y', 'anchor_z',
                 'bb_min_x', 'bb_min_y', 'bb_min_z',
                 'bb_max_x', 'bb_max_y', 'bb_max_z',
                 'n_pixels']

    # we need to switch from our axis conventions (zyx)
    # to java conventions (xyz)
    res_in_micron = resolution[::-1]
    # reshuffle the attributes to fit the output colomns

    def translate_coordinate_tuple(coords):
        coords = coords[::-1]
        for d in range(3):
            coords[:, d] *= res_in_micron[d]
        return coords

    # center of mass / anchor points
    com = translate_coordinate_tuple(attributes[:, 2:5])
    # attributes[5:8] = min coordinate of bounding box
    minc = translate_coordinate_tuple(attributes[:, 5:8])
    # attributes[8:] = min coordinate of bounding box
    maxc = translate_coordinate_tuple(attributes[:, 8:11])

    # NOTE: attributes[1] = size in pixel
    # make the output attributes
    data = np.concatenate([label_ids, com, minc, maxc, attributes[:, 1:2]], axis=1)
    assert data.shape[1] == len(col_names)

    # write to csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(col_names)
        writer.writerows(data)


def base_attributes(input_path, input_key, output_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=True):

    # prepare cluster tools tasks
    make_config(tmp_folder)

    # make base attributes as n5 dataset
    tmp_path, tmp_key = n5_attributes(input_path, input_key,
                                      tmp_folder, target, max_jobs)

    # correct anchor positions
    if correct_anchors:
        pass
        # TODO need to test this first
        # run_correction(input_path, input_key, tmp_path, tmp_key,
        #                tmp_folder, target, max_jobs)

    # write output to csv
    to_csv(tmp_path, tmp_key, output_path, resolution)
