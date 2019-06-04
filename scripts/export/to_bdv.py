import os
import json
import luigi

import z5py
from shutil import rmtree
from cluster_tools.downscaling import PainteraToBdvWorkflow


def check_max_id(path, key):
    with z5py.File(path) as f:
        attrs = f[key].attrs
        max_id = attrs['maxId']
    if max_id > 32000:
        print("Max-id:", max_id, "does not fit int16")
        raise RuntimeError("Uint16 overflow")
    else:
        print("Max-id:", max_id, "fits int16")


def to_bdv(in_path, in_key, out_path, resolution, target='slurm'):
    check_max_id(in_path, in_key)
    tmp_folder = 'tmp_export_bdv'

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    configs = PainteraToBdvWorkflow.get_config()

    global_conf = configs['global']
    global_conf.update({'shebang':
                        "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python",
                        'block_shape': [64, 512, 512]})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    config = configs['copy_volume']
    config.update({'threads_per_job': 8, 'mem_limit': 32, 'time_limit': 1600,
                   'chunks': [32, 256, 256]})
    with open(os.path.join(config_folder, 'copy_volume.config'), 'w') as f:
        json.dump(config, f)

    metadata = {'unit': 'micrometer', 'resolution': resolution}
    task = PainteraToBdvWorkflow(tmp_folder=tmp_folder, max_jobs=1,
                                 config_dir=config_folder, target=target,
                                 input_path=in_path, input_key_prefix=in_key,
                                 output_path=out_path, metadata_dict=metadata,
                                 skip_existing_levels=False, dtype='uint16')
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("Segmentation export failed")
    rmtree(tmp_folder)
