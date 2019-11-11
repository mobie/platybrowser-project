import os
import json
import luigi
import z5py

from cluster_tools.downscaling import DownscalingWorkflow
from paintera_tools import serialize_from_commit, postprocess
from paintera_tools import set_default_shebang as set_ptools_shebang
from paintera_tools import set_default_block_shape as set_ptools_block_shape
from .to_bdv import to_bdv, check_max_id
from .map_segmentation_ids import map_segmentation_ids
from ..default_config import write_default_global_config, get_default_shebang
from ..files import get_postprocess_dict


def get_scale_factors(paintera_path, paintera_key):
    f = z5py.File(paintera_path)
    g = f[paintera_key]['data']
    keys = [int(k[1:]) for k in g.keys()]
    keys = sorted(keys)
    scale_factors = [[1, 1, 1]]
    rel_scales = [[1, 1, 1]]
    for k in keys[1:]:
        factor = g['s%i' % k].attrs['downsamplingFactors']
        rel_factor = [int(sf // prev) for sf, prev in zip(factor, scale_factors[-1])]
        scale_factors.append(factor)
        rel_scales.append(rel_factor[::-1])
    return rel_scales[1:]


def downscale(path, in_key, out_key,
              scale_factors, tmp_folder, max_jobs, target):
    task = DownscalingWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_folder)
    configs = task.get_config()

    config = configs['downscaling']
    config.update({'mem_limit': 8, 'time_limit': 120,
                   'library_kwargs': {'order': 0}})
    with open(os.path.join(config_folder, 'downscaling.config'), 'w') as f:
        json.dump(config, f)

    n_scales = len(scale_factors)
    halos = [[0, 0, 0]] * n_scales

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key,
             output_key_prefix=out_key,
             scale_factors=scale_factors, halos=halos)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling the segmentation failed")


def export_segmentation(paintera_path, paintera_key, folder, new_folder, name, resolution,
                        tmp_folder, target='slurm', max_jobs=200):
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

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    tmp_key = 'seg'
    tmp_key0 = os.path.join(tmp_key, 's0')

    # we need to have the same processing block shape
    # as the paintera labels chunk size, otherwise we
    # will run into issues for label multisets
    with z5py.File(paintera_path, 'r') as f:
        ds = f[os.path.join(paintera_key, 'data', 's0')]
        chunks = ds.chunks

    # set correct shebang and block shape for paintera tools
    set_ptools_shebang(get_default_shebang())
    set_ptools_block_shape(chunks)

    # run post-processing if specified for this segmentation name
    pp_dict = get_postprocess_dict()
    run_postprocessing = name in pp_dict
    if run_postprocessing:
        pp_config = pp_dict[name]
        boundary_path = pp_config['boundary_path']
        boundary_key = pp_config['boundary_key']

        min_segment_size = pp_config.get('min_segment_size', None)
        max_segment_number = pp_config.get('max_segment_number', None)

        label_segmentation = pp_config['label_segmentation']
        tmp_postprocess = os.path.join(tmp_folder, 'postprocess_paintera')
        postprocess(paintera_path, paintera_key,
                    boundary_path, boundary_key,
                    tmp_folder=tmp_postprocess,
                    target=target, max_jobs=max_jobs,
                    n_threads=16, size_threshold=min_segment_size,
                    target_number=max_segment_number,
                    label=label_segmentation,
                    output_path=tmp_path, output_key=tmp_key0)

    else:
        if name == 'sbem-6dpf-1-whole-segmented-cilia-labels':
            map_to_background = [1]
        else:
            map_to_background = None
        # export segmentation from paintera commit for all scales
        serialize_from_commit(paintera_path, paintera_key, tmp_path, tmp_key0, tmp_folder,
                              max_jobs, target, relabel_output=True, map_to_background=map_to_background)

    # check for overflow
    print("Check max-id @", tmp_path, tmp_key0)
    check_max_id(tmp_path, tmp_key0)

    # downscale the segemntation
    scale_factors = get_scale_factors(paintera_path, paintera_key)
    downscale(tmp_path, tmp_key0, tmp_key, scale_factors, tmp_folder, max_jobs, target)

    # convert to bdv
    out_path = os.path.join(new_folder, 'segmentations', '%s.h5' % name)
    tmp_bdv = os.path.join(tmp_folder, 'tmp_bdv')
    to_bdv(tmp_path, tmp_key, out_path, resolution, tmp_bdv, target)

    # compute mapping to old segmentation
    # this can be skipped for new segmentations by setting folder to None
    if folder is not None:
        map_segmentation_ids(folder, new_folder, name, tmp_folder, max_jobs, target)
