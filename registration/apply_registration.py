#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import argparse
import os
import json
import random
from shutil import rmtree

import luigi
from scripts.extension.registration import ApplyRegistrationLocal, ApplyRegistrationSlurm


def apply_registration(input_path, output_path, transformation_file,
                       interpolation='nearest', output_format='tif', result_dtype='unsigned char',
                       target='local'):
    task = ApplyRegistrationSlurm if target == 'slurm' else ApplyRegistrationLocal
    assert result_dtype in task.result_types
    assert interpolation in task.interpolation_modes

    rand_id = hash(random.uniform(0, 1000000))
    tmp_folder = 'tmp_%i' % rand_id
    config_dir = os.path.join(tmp_folder, 'configs')

    os.makedirs(config_dir, exist_ok=True)

    shebang = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python'
    conf = task.default_global_config()
    conf.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    task_config = task.default_task_config()
    task_config.update({'mem_limit': 16, 'time_limit': 240, 'threads_per_job': 4,
                        'ResultImagePixelType': result_dtype})
    with open(os.path.join(config_dir, 'apply_registration.config'), 'w') as f:
        json.dump(task_config, f)

    in_file = os.path.join(tmp_folder, 'inputs.json')
    with open(in_file, 'w') as f:
        json.dump([input_path], f)

    out_file = os.path.join(tmp_folder, 'outputs.json')
    with open(out_file, 'w') as f:
        json.dump([output_path], f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=1,
             input_path_file=in_file, output_path_file=out_file,
             transformation_file=transformation_file, output_format=output_format,
             interpolation=interpolation)
    ret = luigi.build([t], local_scheduler=True)

    if not ret:
        raise RuntimeError("Apply registration failed")

    rmtree(tmp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply registration to input tif file')
    parser.add_argument('input_path', type=str,
                        help="Path to input image volume. Must be tiff and have resolution information.")
    parser.add_argument('output_path', type=str, help="Path to output.")
    parser.add_argument('transformation_file', type=str, help="Path to transformation to apply.")
    parser.add_argument('--interpolation', type=str, default='nearest',
                        help="Interpolation order that will be used. Can be 'nearest' or 'linear'.")
    parser.add_argument('--output_format', type=str, default='tif',
                        help="Output file format. Can be 'tif' or 'xml'.")
    parser.add_argument('--result_dtype', type=str, default='unsigned char',
                        help="Image datatype. Can be 'unsigned char' or 'unsigned short'.")
    parser.add_argument('--target', type=str, default='local',
                        help="Where to run the computation. Can be 'local' or 'slurm'.")

    args = parser.parse_args()
    apply_registration(args.input_path, args.output_path, args.transformation_file,
                       args.interpolation, args.output_format, args.result_dtype,
                       args.target)
