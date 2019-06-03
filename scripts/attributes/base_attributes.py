#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python
# TODO new platy-browser env

import os
import json
from shutil import rmtree

import luigi
from cluster_tools.morphology import MorphologyWorkflow


def n5_attributes(input_path, input_key,
                  targt='slurm', max_jobs=100):
    task = MorphologyWorkflow

    tmp_folder = './tmp_attributes'
    config_folder = os.path.join(tmp_folder, configs)
    out_path = os.path.join(tmp_folder, 'data.n5')

    configs = task.get_config()
    os.makedirs(config_folder, exist_ok=True)

    global_config = configs['global']

    # TODO use new platy browser env
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_config['shebang'] = shebang
    global_config['block_shape'] = [64, 512, 512]
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    out_key = 'attributes'
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs, target=target,
             config_dir=config_folder,
             input_path=path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix='attributes', max_jobs_merge=32)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Attribute workflow failed")
    return out_path, out_key


def csv_attributes(input_path, input_key, output_path):
    pass


# TODO should probably do this on downscaled version of segmentation
# correct anchor points that are not inside of objects
def run_correction(input_path, input_key, output_path):
    pass


def clean_up():
    rmtree('./tmp_attributes')


def base_attributes(input_path, input_key, output_path, correct_anchors=True):
    tmp_path, tmp_key = n5_attributes(input_path, input_key)
    csv_attributes(tmp_path, tmp_key, output_path)
    if correct_anchors:
        run_correction(input_path, input_key, output_path)
    # clean_up()
