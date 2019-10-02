import os
import json
import luigi
import pandas as pd

from elf.io import open_file
from cluster_tools.transformations import LinearTransformationWorkflow
from cluster_tools.downscaling import DownscalingWorkflow, PainteraToBdvWorkflow
from ..default_config import write_default_global_config


def csv_to_json(trafo_path):
    trafo = pd.read_csv(trafo_path, sep='\t')
    n_slices = trafo.shape[0]
    trafo = {z: {'a': trafo.loc[z].mult, 'b': trafo.loc[z].offset} for z in range(n_slices)}
    new_trafo_path = os.path.splitext(trafo_path)[0] + '.json'
    with open(new_trafo_path, 'w') as f:
        json.dump(trafo, f)
    return new_trafo_path


def validate_trafo(trafo_path, in_path, in_key):
    with open_file(in_path, 'r') as f:
        n_slices = f[in_key].shape[0]
    with open(trafo_path) as f:
        trafo = json.load(f)
    if n_slices != len(trafo):
        raise RuntimeError("Invalid number of transformations: %i,%i" % (n_slices, len(trafo)))


def downsample(ref_path, in_path, in_key, out_path, resolution,
               tmp_folder, target, max_jobs):
    with open_file(ref_path, 'r') as f:
        g = f['t00000/s00']
        levels = list(g.keys())
        levels.sort()

        sample_factors = []
        for level in range(1, len(levels)):
            ds0 = g['%s/cells' % levels[level - 1]]
            ds1 = g['%s/cells' % levels[level]]

            s0 = ds0.shape
            s1 = ds1.shape
            factor = [int(round(float(sh0) / sh1, 0)) for sh0, sh1 in zip(s0, s1)]

            sample_factors.append(factor)
        assert len(sample_factors) == len(levels) - 1

    config_dir = os.path.join(tmp_folder, 'configs')
    task = DownscalingWorkflow
    config = task.get_config()['downscaling']
    config.update({'library': 'skimage', 'time_limit': 240, 'mem_limit': 4})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(config, f)

    tmp_key2 = 'downscaled'
    halos = len(sample_factors) * [[0, 0, 0]]

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=in_path, input_key=in_key,
             scale_factors=sample_factors, halos=halos,
             metadata_format='paintera', metadata_dict={'resolution': resolution},
             output_path=in_path, output_key_prefix=tmp_key2)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling failed")

    task = PainteraToBdvWorkflow
    config = task.get_config()['copy_volume']
    config.update({'threads_per_job': 32, 'mem_limit': 64, 'time_limit': 1400})
    with open(os.path.join(config_dir, 'copy_volume.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=1, target=target,
             input_path=in_path, input_key_prefix=tmp_key2,
             output_path=out_path,
             metadata_dict={'resolution': resolution})
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling failed")


def intensity_correction(in_path, out_path, mask_path, mask_key,
                         trafo_path, tmp_folder, resolution,
                         target='slurm', max_jobs=250, tmp_path=None):
    trafo_ext = os.path.splitext(trafo_path)[1]
    if trafo_ext == '.csv':
        trafo_path = csv_to_json(trafo_path)
    elif trafo_ext != '.json':
        raise ValueError("Expect trafo as json.")

    key = 't00000/s00/0/cells'
    validate_trafo(trafo_path, in_path, key)

    config_dir = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_dir)

    if tmp_path is None:
        tmp_path = os.path.join(tmp_folder, 'data.n5')
    tmp_key = 'data'

    task = LinearTransformationWorkflow
    conf = task.get_config()['linear']
    conf.update({'time_limit': 600, 'mem_limit': 4})
    with open(os.path.join(config_dir, 'linear.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=in_path, input_key=key,
             mask_path=mask_path, mask_key=mask_key,
             output_path=tmp_path, output_key=tmp_key,
             transformation=trafo_path)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Transformation failed")

    downsample(in_path, tmp_path, tmp_key, out_path, resolution,
               tmp_folder, target, max_jobs)
