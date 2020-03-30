import os
import json
import luigi
import z5py

from cluster_tools.node_labels import NodeLabelWorkflow
from pybdv.metadata import get_data_path
from pybdv.util import get_key
from ..default_config import write_default_global_config
from ..util import is_h5_file


def get_seg_path(folder, name):
    # check if we have a data sub folder, if we have it load
    # the segmentation from there
    data_folder = os.path.join(folder, 'images', 'local')
    data_folder = data_folder if os.path.exists(data_folder) else folder

    # check if we have an xml
    path = os.path.join(data_folder, '%s.xml' % name)
    # read h5 path from the xml
    if os.path.exists(path):
        path = get_data_path(path, return_absolute_path=True)
        if not os.path.exists(path):
            raise RuntimeError("Invalid path in xml")
        return path
    else:
        raise RuntimeError("The specified folder does not contain segmentation file with name %s" % name)


def map_ids(path1, path2, out_path, tmp_folder, max_jobs, target, prefix,
            key1=None, key2=None, scale=0):
    task = NodeLabelWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_folder)
    configs = task.get_config()

    conf = configs['merge_node_labels']
    conf.update({'threads_per_job': 8, 'mem_limit': 16})
    with open(os.path.join(config_folder, 'merge_node_labels.config'), 'w') as f:
        json.dump(conf, f)

    if key1 is None:
        is_h5 = is_h5_file(path1)
        key1 = get_key(is_h5, time_point=0, setup_id=0, scale=scale)
    if key2 is None:
        is_h5 = is_h5_file(path2)
        key2 = get_key(is_h5, time_point=0, setup_id=0, scale=scale)

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    tmp_key = prefix
    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             ws_path=path1, ws_key=key1,
             input_path=path2, input_key=key2,
             output_path=tmp_path, output_key=tmp_key,
             prefix=prefix, max_overlap=True, serialize_counts=True)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Id-mapping failed")

    ds = z5py.File(tmp_path)[tmp_key]
    lut = ds[:]
    assert lut.ndim == 2
    lut = dict(zip(range(len(lut)), lut.tolist()))

    with open(out_path, 'w') as f:
        json.dump(lut, f)


def map_segmentation_ids(src_folder, dest_folder, name, tmp_folder, max_jobs, target):
    # might not have an initial version of the segmentation and in this case need to skip
    try:
        src_path = get_seg_path(src_folder, name)
    except RuntimeError:
        print("Did not find old segmentation dataset for %s in %s" % (src_folder, name))
        print("Skip mapping of segmentation ids")
        return
    dest_path = get_seg_path(dest_folder, name)

    # map ids from src to dest via maximal overlap
    out_path = os.path.join(dest_folder, 'misc', 'new_id_lut_%s.json' % name)
    map_ids(src_path, dest_path, out_path, tmp_folder, max_jobs, target,
            prefix='to_dest')
