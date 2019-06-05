import os
import json
import luigi
import z5py

from cluster_tools.downscaling import DownscalingWorkflow
from paintera_tools import serialize_from_commit
from .to_bdv import to_bdv
from .map_segmentation_ids import map_segmentation_ids


def get_n_scales(paintera_path, paintera_key):
    f = z5py.File(paintera_path)
    g = f[paintera_key]['data']
    keys = list(g.keys())
    scales = [key for key in keys
              if os.path.isdir(os.path.join(g.path, key)) and key.startswith('s')]
    return len(scales)


def downscale(path, in_key, out_key,
              n_scales, tmp_folder, max_jobs, target):
    task = DownscalingWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    configs = task.get_config()

    global_conf = configs['global']
    global_conf.update({'shebang':
                        "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python",
                        'block_shape': [64, 512, 512]})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    config = configs['downscaling']
    config.update({'mem_limit': 8, 'time_limit': 120,
                   'library_kwargs': {'order': 0}})
    with open(os.path.join(config_folder, 'downscaling.config'), 'w') as f:
        json.dump(config, f)

    # for now we hard-code scale factors to 2, but it would be cleaner to infer this from the data
    scales = [[2, 2, 2]] * n_scales
    halos = [[0, 0, 0]] * n_scales

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key,
             output_key_prefix=out_key,
             scale_factors=scales, halos=halos)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling the segmentation failed")


def export_segmentation(paintera_path, paintera_key, folder, new_folder, name, resolution, tmp_folder):
    """ Export a segmentation from paintera project to bdv file and
    compute segment lut for previous segmentation.

    Arguments:
        paintera_path: path to the paintera project corresponding to the new segmentation
        paintera_key: key to the paintera project corresponding to the new segmentation
        folder: folder for old segmentation
        new_folder: folder for new segmentation
        name: name of segmentation
        resolution: resolution [z, y, x] in micrometer
        tmp_folder: folder for temporary files
    """
    # TODO should make this a param
    max_jobs = 250
    target = 'slurm'

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    tmp_key = 'seg'
    tmp_key0 = os.path.join(tmp_key, 's0')

    # export segmentation from paintera commit for all scales
    serialize_from_commit(paintera_path, paintera_key, tmp_path, tmp_key0, tmp_folder,
                          max_jobs, target, relabel_output=True)

    # TODO run small size filter postprocessing ?

    # downscale the segemntation
    n_scales = get_n_scales(paintera_path, paintera_key)
    downscale(tmp_path, tmp_key0, tmp_key, n_scales, tmp_folder, max_jobs, target)

    # convert to bdv
    out_path = os.path.join(new_folder, 'segmentations', '%s.h5' % name)
    tmp_bdv = os.path.join(tmp_folder, 'tmp_bdv')
    to_bdv(tmp_path, tmp_key, out_path, resolution, tmp_bdv, target)

    # compute mapping to old segmentation
    map_segmentation_ids(folder, new_folder, name, tmp_folder, max_jobs, target)
