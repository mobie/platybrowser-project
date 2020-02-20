import os
import json
import luigi

from cluster_tools.downscaling import DownscalingWorkflow
from mmpb.default_config import get_default_shebang


def downsample_for_resin(target, max_jobs):
    task = DownscalingWorkflow

    path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    in_key = 'setup0/timepoint0/s1'
    out_path = '../data.n5'
    out_key = 'volumes/volumes/raw-samplexy'

    scale_factors = [[1, 4, 4], [1, 2, 2], [1, 2, 2], [1, 2, 2]]
    halos = [[0, 0, 0]] * len(scale_factors)

    tmp_folder = 'tmp_downsample'
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    configs = task.get_config()
    global_conf = configs['global']
    block_shape = [256, 128, 128]
    shebang = get_default_shebang()
    global_conf.update({'block_shape': block_shape, 'shebang': shebang})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    conf = configs['downscaling']
    conf.update({'library': 'skimage',
                 'library_kwargs': {'function': 'mean'}})

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key,
             output_path=out_path, output_key_prefix=out_key,
             scale_factors=scale_factors, halos=halos)
    luigi.build([t], local_scheduler=True)


if __name__ == '__main__':
    downsample_for_resin('local', 64)
