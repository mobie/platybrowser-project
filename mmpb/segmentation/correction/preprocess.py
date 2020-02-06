import os
import json
import luigi
from elf.io import open_file

from paintera_tools import serialize_from_commit
from paintera_tools.util import compute_graph_and_weights
from cluster_tools.node_labels import NodeLabelWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.downscaling import DownscalingWorkflow


def graph_and_costs(path, ws_key, aff_key, out_path):
    tmp_folder = './tmp_preprocess'
    compute_graph_and_weights(path, aff_key,
                              path, ws_key, out_path,
                              tmp_folder, target='slurm', max_jobs=250,
                              offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                              with_costs=True)


def accumulate_node_labels(ws_path, ws_key, seg_path, seg_key,
                           out_path, out_key, prefix):
    task = NodeLabelWorkflow

    tmp_folder = './tmp_preprocess'
    config_dir = os.path.join(tmp_folder, 'configs')

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target='slurm', max_jobs=250,
             ws_path=ws_path, ws_key=ws_key,
             input_path=seg_path, input_key=seg_key,
             output_path=out_path, output_key=out_key,
             prefix=prefix)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def compute_bounding_boxes(path, key):
    task = MorphologyWorkflow
    tmp_folder = './tmp_preprocess'
    config_dir = os.path.join(tmp_folder, 'configs')

    out_key = 'morphology'
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target='slurm', max_jobs=250,
             input_path=path, input_key=key,
             output_path=path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def downscale_segmentation(path, key):
    task = DownscalingWorkflow

    tmp_folder = './tmp_preprocess'
    config_dir = os.path.join(tmp_folder, 'configs')

    configs = task.get_config()
    conf = configs['downscaling']
    conf.update({'library_kwargs': {'order': 0}})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(conf, f)

    in_key = os.path.join(key, 's0')
    n_scales = 5
    scales = n_scales * [[2, 2, 2]]
    halos = n_scales * [[0, 0, 0]]

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             # target='slurm', max_jobs=250,
             target='local', max_jobs=64,
             input_path=path, input_key=in_key,
             scale_factors=scales, halos=halos,
             output_path=path, output_key_prefix=key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def copy_tissue_labels(in_path, out_path, out_key):
    with open_file(in_path, 'r') as f:
        names = f['semantic_names'][:]
        mapping = f['semantic_mapping'][:]

    semantics = {name: ids.tolist() for name, ids in zip(names, mapping)}
    with open_file(out_path) as f:
        ds = f[out_key]
        ds.attrs['semantics'] = semantics


# preprocess:
# - export current paintera segmentation
# - build graph and compute weights for current superpixels
# - get current node labeling
# - compute bounding boxes for current segments
# - downscale the segmentation
def preprocess(path, key, aff_key,
               tissue_path, tissue_key,
               out_path, out_key):
    tmp_folder = './tmp_preprocess'
    out_key0 = os.path.join(out_key, 's0')

    serialize_from_commit(path, key, out_path, out_key0, tmp_folder,
                          max_jobs=250, target='slurm', relabel_output=True)

    ws_key = os.path.join(key, 'data', 's0')
    graph_and_costs(path, ws_key, aff_key, out_path)

    accumulate_node_labels(path, ws_key, out_path, out_key0,
                           out_path, 'node_labels', prefix='node_labels')

    accumulate_node_labels(out_path, out_key0, tissue_path, tissue_key,
                           out_path, 'tissue_labels', prefix='tissue')
    copy_tissue_labels(tissue_path, out_path, 'tissue_labels')

    compute_bounding_boxes(out_path, out_key0)
    downscale_segmentation(out_path, out_key)
