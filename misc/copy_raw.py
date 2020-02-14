import os
import json
import luigi
from cluster_tools.copy_volume import CopyVolumeSlurm, CopyVolumeLocal
from cluster_tools.downscaling import DownscalingWorkflow
from pybdv.util import get_scale_factors


# copy the raw data - only scale 0
# we don't copy the other scales, because there are some
# artifacts, so we do the downscalig again
def copy_raw_base(target, max_jobs):
    task = CopyVolumeSlurm if target == 'slurm' else CopyVolumeLocal
    in_file = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    in_key = 't00000/s00/0/cells'
    out_file = '../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    out_key = 'setup0/timepoint0/s0'
    chunks = (96,) * 3

    tmp_folder = 'tmp_copy'
    config_dir = 'tmp_copy/configs'
    os.makedirs(config_dir, exist_ok=True)

    global_conf = task.default_global_config()
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_conf.update({'shebang': shebang, 'block_shape': chunks})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    task_conf = task.default_task_config()
    task_conf.update({'chunks': chunks, 'time_limit': 900, 'mem_limit': 2})
    with open(os.path.join(config_dir, 'copy_volume.config'), 'w') as f:
        json.dump(task_conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
             input_path=in_file, input_key=in_key,
             output_path=out_file, output_key=out_key,
             prefix='raw')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def get_ds_factors():
    in_file = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    abs_scale_factors = get_scale_factors(in_file, 0)[1:]
    scale_factors = [[int(sf) for sf in abs_scale_factors[0]]]
    for ii in range(1, len(abs_scale_factors)):
        rel_sf = [int(sf1 / sf2) for sf1, sf2 in zip(abs_scale_factors[ii],
                                                     abs_scale_factors[ii - 1])]
        scale_factors.append(rel_sf)
    return scale_factors


# downscale raw
def downsacle_raw(target, max_jobs):
    task = DownscalingWorkflow
    # determine scales and downscaling factors we need
    scale_factors = get_ds_factors()
    halos = len(scale_factors) * [[2, 2, 2]]

    path = '../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    in_key = 'setup0/timepoint0/s0'
    chunks = (96,) * 3

    tmp_folder = 'tmp_downscale'
    config_dir = 'tmp_downscale/configs'
    os.makedirs(config_dir, exist_ok=True)

    configs = task.get_config()
    global_conf = configs['global']
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_conf.update({'shebang': shebang, 'block_shape': chunks})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    task_conf = configs['downscaling']
    task_conf.update({'chunks': chunks, 'time_limit': 900, 'mem_limit': 2,
                      'library': 'skimage', 'library_kwargs': {'function': 'mean'}})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(task_conf, f)

    metadata = {'resolution': [0.025, 0.01, 0.01], 'unit': 'micrometer'}
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=in_key,
             output_path=path, scale_factors=scale_factors,
             halos=halos, metadata_format='bdv.n5',
             metadata_dictt=metadata)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


if __name__ == '__main__':
    # copy_raw_base('slurm', 200)
    downsacle_raw('local', 48)
