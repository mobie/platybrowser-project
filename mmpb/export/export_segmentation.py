import os
import json
import luigi
import numpy as np
import z5py

from cluster_tools.downscaling import DownscalingWorkflow
from paintera_tools import serialize_from_commit, postprocess
from paintera_tools import set_default_shebang as set_ptools_shebang
from paintera_tools import set_default_block_shape as set_ptools_block_shape
from .map_segmentation_ids import map_segmentation_ids
from ..default_config import write_default_global_config, get_default_shebang, get_default_block_shape
from ..util import add_max_id


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


def get_scale_factors_from_paintera(paintera_path, paintera_key):
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


def downscale(path, scale_factors, resolution, max_id, tmp_folder, max_jobs, target):
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

    metadata = {'resolution': resolution, 'unit': 'micrometer'}
    in_key = 'setup0/timepoint0/s0'
    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key,
             scale_factors=scale_factors, halos=halos,
             metadata_format='bdv.n5', metadata_dict=metadata)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling the segmentation failed")

    for scale in range(n_scales + 1):
        scale_key = 'setup0/timepoint0/s%i' % scale
        add_max_id(path, scale_key, max_id=max_id)


def export_segmentation(paintera_path, paintera_key, name,
                        folder, new_folder, out_path, resolution, tmp_folder,
                        pp_config=None, map_to_background=None, chunks=None,
                        target='slurm', max_jobs=200):
    """ Export a segmentation from paintera project to bdv file and
    compute segment lut for previous segmentation.

    Arguments:
        paintera_path: path to the paintera project corresponding to the new segmentation
        paintera_key: key to the paintera project corresponding to the new segmentation
        name: name of the segmentation
        folder: folder for old segmentation
        new_folder: folder for new segmentation
        out_path: output path for the exported segmentation
        resolution: resolution of the data
        tmp_folder: folder for temporary files
        pp_config: config for segmentation post-processing (default: None)
        map_to_background: additional ids that shall be mapped to background / 0 (default: None)
        chunks: chunks used for serialization (default: None)
        target: computation target (default: 'slurm')
        max_jobs: maximal number of jobs used for computation (default: 200)
    """

    # TODO need to implement different output chunk size for label multiset!
    with z5py.File(paintera_path, 'r') as f:
        ds = f[os.path.join(paintera_key, 'data', 's0')]
        is_label_multiset = ds.attrs.get("isLabelMultiset", False)
        if chunks is None:
            chunks = ds.chunks
        elif chunks is not None and is_label_multiset:
            raise NotImplementedError("""Chunk size different from the dataset
                                         chunks is not implemented for label multiset data""")

    # set correct shebang and block shape for paintera tools
    set_ptools_shebang(get_default_shebang())
    set_ptools_block_shape(get_default_block_shape())

    out_key = 'setup0/timepoint0/s0'
    # run post-processing if specified for this segmentation name
    if pp_config is not None:
        boundary_path = pp_config['BoundaryPath']
        boundary_key = pp_config['BoundaryKey']

        min_segment_size = pp_config.get('MinSegmentSize', None)
        max_segment_number = pp_config.get('MaxSegmentNumber', None)

        label_segmentation = pp_config['LabelSegmentation']
        tmp_postprocess = os.path.join(tmp_folder, 'postprocess_paintera')
        postprocess(paintera_path, paintera_key,
                    boundary_path, boundary_key,
                    tmp_folder=tmp_postprocess,
                    target=target, max_jobs=max_jobs,
                    n_threads=16, size_threshold=min_segment_size,
                    target_number=max_segment_number,
                    label=label_segmentation,
                    output_path=out_path, output_key=out_key)

    else:
        # export segmentation from paintera commit for all scales
        serialize_from_commit(paintera_path, paintera_key, out_path, out_key, tmp_folder,
                              max_jobs, target, relabel_output=True, map_to_background=map_to_background)

    # check for overflow
    # now that we can export to n5, we don't really need this check any more,
    # still leaving it here for now to stay consistent with old versions
    # this is not really necessary an more, because bdv-n5 supports uint32 / uint64 now.
    # still leaving it here to be compatible with legacy code
    print("Check max-id @", out_path, out_key)
    max_id = check_max_id(out_path, out_key)

    # downscale the segemntation
    scale_factors = get_scale_factors_from_paintera(paintera_path, paintera_key)
    downscale(out_path, scale_factors, resolution, max_id, tmp_folder, max_jobs, target)

    # compute mapping to old segmentation
    # this can be skipped for new segmentations by setting folder to None
    if folder is not None:
        map_segmentation_ids(folder, new_folder, name, tmp_folder, max_jobs, target)
