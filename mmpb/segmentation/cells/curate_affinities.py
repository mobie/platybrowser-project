import os
import json
import luigi
from cluster_tools.affinities import InsertAffinitiesWorkflow
from mmpb.default_config import get_default_block_shape, get_default_shebang


def curate_affinities(path, in_key, out_key,
                      region_path, region_key,
                      tmp_folder, target, max_jobs,
                      roi_begin=None, roi_end=None):

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    configs = InsertAffinitiesWorkflow.get_config()

    conf = configs['global']
    shebang = get_default_shebang()
    block_shape = get_default_block_shape()
    conf.update({'shebang': shebang, 'block_shape': block_shape,
                 'roi_begin': roi_begin, 'roi_end': roi_end})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['insert_affinities']
    # make sure that this is the only cuticle and neuropil label
    ignore_ids = [1, 2]
    chunks = [1] + [bs // 2 for bs in block_shape]
    conf.update({'time_limit': 120, 'mem_limit': 6, 'chunks': chunks,
                 'zero_objects_list': ignore_ids, 'erode_by': 3, 'dilate_by': 2})
    with open(os.path.join(config_folder, 'insert_affinities.config'), 'w') as f:
        json.dump(conf, f)

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    t = InsertAffinitiesWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                                 max_jobs=max_jobs, target=target,
                                 input_path=path, input_key=in_key,
                                 output_path=path, output_key=out_key,
                                 objects_path=region_path, objects_key=region_key,
                                 offsets=offsets)
    ret = luigi.build([t], local_scheduler=True)
    if ret:
        raise RuntimeError("Affinity curation failed")
