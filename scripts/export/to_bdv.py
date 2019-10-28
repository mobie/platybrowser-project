import os
import json
import luigi

import numpy as np
import h5py
import z5py
from cluster_tools.downscaling import PainteraToBdvWorkflow
from ..default_config import write_default_global_config


def check_max_id(path, key):
    with z5py.File(path) as f:
        attrs = f[key].attrs
        max_id = attrs['maxId']
    if max_id > np.iinfo('int16').max:
        print("Max-id:", max_id, "does not fit int16")
        raise RuntimeError("Uint16 overflow")
    else:
        print("Max-id:", max_id, "fits int16")
    return max_id


def to_bdv(in_path, in_key, out_path, resolution, tmp_folder, target='slurm'):
    max_id = check_max_id(in_path, in_key)

    config_folder = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_folder)
    configs = PainteraToBdvWorkflow.get_config()

    config = configs['copy_volume']
    config.update({'threads_per_job': 32, 'mem_limit': 64, 'time_limit': 2400,
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

    # write the max-id for all datasets
    with h5py.File(out_path) as f:
        g = f['t00000/s00']
        for scale, scale_group in g.items():
            ds = scale_group['cells']
            ds.attrs['maxId'] = max_id
