import os
import json
import luigi
import numpy as np
import z5py
from elf.io import open_file

from paintera_tools import serialize_from_commit, set_default_roi
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


def compute_bounding_boxes(path, key, out_key,
                           tmp_folder, target, max_jobs):
    task = MorphologyWorkflow
    config_dir = os.path.join(tmp_folder, 'configs')

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
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


def write_flagged_ids(out_folder, flagged_ids):
    assert isinstance(flagged_ids, list)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, 'flagged_ids.json')
    with open(out_path, 'w') as f:
        json.dump(flagged_ids, f)
    return out_path


def write_out_file(out_folder, flagged_ids,
                   raw_path, raw_key,
                   ws_path, ws_key,
                   node_label_path, node_label_key,
                   table_path, table_key,
                   problem_path, graph_key, feat_key,
                   scale_factor, flagged_id_path):
    os.makedirs(out_folder, exist_ok=True)
    conf_path = os.path.join(out_folder, 'correct_false_merges_config.json')
    conf = {'raw_path': raw_path, 'raw_key': raw_key,
            'ws_path': ws_path, 'ws_key': ws_key,
            'node_label_path': node_label_path, 'node_label_key': node_label_key,
            'table_path': table_path, 'table_key': table_key,
            'problem_path': problem_path, 'graph_key': graph_key, 'feat_key': feat_key,
            'scale_factor': scale_factor, 'false_merge_id_path': flagged_id_path}
    with open(conf_path, 'w') as f:
        json.dump(conf, f)


def resave_assignements(assignments, path, out_key):
    node_ids = assignments[:, 0]
    node_labels = assignments[:, 1]
    n_nodes = int(node_ids.max()) + 1
    full_node_labels = np.zeros(n_nodes, dtype='uint64')
    full_node_labels[node_ids] = node_labels

    with z5py.File(path) as f:
        ds = f.require_dataset(out_key, shape=full_node_labels.shape, compression='gzip',
                               chunks=full_node_labels.shape, dtype='uint64')
        ds[:] = full_node_labels


def preprocess_from_paintera_project(project_path, out_folder,
                                     raw_path, raw_root_key,
                                     boundaries_path, boundaries_key,
                                     out_key, preprocess_scale, work_scale,
                                     tmp_folder, target, max_jobs,
                                     roi_begin=None, roi_end=None):
    """ Run all pre-processing necessary for the correction tool
    based on segemntation in a paintera project.
    """

    # paintera has two diffe
    source_name = 'org.janelia.saalfeldlab.paintera.state.label.ConnectomicsLabelState'

    # parse the paintera attribues.json to extract the relevant paths + locked ids
    attrs_path = os.path.join(project_path, 'attributes.json')
    with open(attrs_path, 'r') as f:
        attrs = json.load(f)

    # get the segmentation path
    sources = attrs['paintera']['sourceInfo']['sources']
    have_seg_source = False
    for source in sources:
        type_ = source['type']
        if type_ == source_name:
            assert not have_seg_source, "Only support a single segmentation source!"
            source_state = source['state']

            seg_data = source_state['backend']['data']
            seg_path = seg_data['container']['data']['basePath']
            seg_root_key = seg_data['dataset']

            # make sure that there are no un-commited actions
            actions = seg_data['fragmentSegmentAssignment']['actions']
            assert len(actions) == 0, "The project state was not properly commited yet, please commit first!"

            flagged_ids = source_state['flaggedSegments']

            have_seg_source = True

    assert have_seg_source, "Did not find any segmentation source"

    # need to set the proper roi for the serialization and graph / weights
    set_default_roi(roi_begin, roi_end)
    # NOTE serialize_from_commit calls the function that writes all the configs
    # serialize the current segmentation
    serialize_from_commit(seg_path, seg_root_key, seg_path, out_key, tmp_folder,
                          max_jobs=max_jobs, target=target, relabel_output=False, scale=preprocess_scale)

    # compute graph and edge weights
    seg_key = os.path.join(seg_root_key, 'data', 's0')
    compute_graph_and_weights(boundaries_path, boundaries_key,
                              seg_path, seg_key, seg_path,
                              tmp_folder, target=target, max_jobs=max_jobs,
                              offsets=None, with_costs=False)

    # save fragment segment assignment in the format the splitting tool can parse
    ass_key = os.path.join(seg_root_key, 'fragment-segment-assignment')
    with open_file(seg_path, 'r') as f:
        assignments = f[ass_key][:].T
    node_label_key = 'node_labels/labels_before_splitting'
    resave_assignements(assignments, seg_path, node_label_key)

    # compute the morphology table for bounding boxes
    table_key = 'morphology'
    compute_bounding_boxes(seg_path, out_key, table_key,
                           tmp_folder, target, max_jobs)

    raw_scale = work_scale + 1
    seg_scale = work_scale

    scale_factor = 2 ** seg_scale
    raw_key = os.path.join(raw_root_key, 's%i' % raw_scale)
    seg_key = os.path.join(seg_root_key, 'data', 's%i' % seg_scale)

    graph_key = 's%i/graph' % preprocess_scale
    feat_key = 'features'
    flagged_id_path = write_flagged_ids(out_folder, flagged_ids)
    write_out_file(out_folder, flagged_ids,
                   raw_path, raw_key,
                   seg_path, seg_key,
                   seg_path, node_label_key,
                   seg_path, table_key,
                   seg_path, graph_key, feat_key,
                   scale_factor, flagged_id_path)


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

    target = 'slurm'
    max_jobs = 200
    compute_bounding_boxes(path, out_key0, 'morphology',
                           tmp_folder, target, max_jobs)
    downscale_segmentation(out_path, out_key)
