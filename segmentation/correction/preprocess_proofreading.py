import os
import json
import luigi
from concurrent import futures

import numpy as np
import nifty
import vigra
import z5py

from cluster_tools.features import EdgeFeaturesWorkflow
from cluster_tools.graph import GraphWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm
from elf.segmentation.clustering import agglomerative_clustering
from sklearn.cluster import AgglomerativeClustering

from mmpb.attributes.util import node_labels
from mmpb.default_config import write_default_global_config

from paintera_tools.serialize.serialize_from_commit import (serialize_assignments,
                                                            serialize_merged_segmentation)
from paintera_tools.util import find_uniques

from common import (PAINTERA_PATH, PAINTERA_KEY,
                    SEG_PATH, SEG_KEY, TMP_PATH,
                    LABEL_MAPPING_PATH, ROI_PATH,
                    BOUNDARY_PATH, BOUNDARY_KEY)


#
# make label division for subprojects
# by assigning labels to blocks
#


def get_blocking(scale, block_shape):
    g = z5py.File(PAINTERA_PATH)[PAINTERA_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape
    blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
    return shape, blocking


def tentative_block_shape(scale, n_target_blocks):
    g = z5py.File(PAINTERA_PATH)[PAINTERA_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape

    size = float(np.prod(shape))
    target_size = size / n_target_blocks

    block_len = int(target_size ** (1. / 3))
    block_len = block_len - (block_len % 64)
    block_shape = 3 * (block_len,)
    _, blocking = get_blocking(scale, block_shape)
    print("Block shape:", block_shape)
    print("Resulting in", blocking.numberOfBlocks, "blocks")
    return block_shape


def make_subdivision_vol(scale, block_shape):
    shape, blocking = get_blocking(scale, block_shape)
    f = z5py.File(TMP_PATH)
    out_key = 'labels_for_subdivision'
    if out_key in f:
        return blocking.numberOfBlocks

    ds = f.require_dataset(out_key, shape=shape, chunks=(64,) * 3, compression='gzip',
                           dtype='uint32')

    def _write_id(block_id):
        print("Write for ", block_id)
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        ds[bb] = block_id + 1

    n_threads = 8
    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_write_id, block_id) for block_id in range(n_blocks)]
        [t.result() for t in tasks]
    return n_blocks


def map_labels_to_blocks(n_blocks, tmp_folder, target, max_jobs):
    block_labels = node_labels(TMP_PATH, 'volumes/segmentation',
                               TMP_PATH, 'labels_for_subdivision', 'for_subdivision',
                               tmp_folder, target=target, max_jobs=max_jobs,
                               max_overlap=True, ignore_label=None)
    labels_to_blocks = {}
    for block_id in range(1, n_blocks + 1):
        this_labels = np.where(block_labels == block_id)[0]
        if len(this_labels) == 0:
            this_labels = []
        elif this_labels[0] == 0:
            this_labels = this_labels[1:].tolist()
        else:
            this_labels = this_labels.tolist()
        labels_to_blocks[block_id] = this_labels
    with open(LABEL_MAPPING_PATH, 'w') as f:
        json.dump(labels_to_blocks, f)
    return labels_to_blocks


def divide_labels_by_blocking(scale, n_target_projects,
                              tmp_folder, target, max_jobs):
    if os.path.exists(LABEL_MAPPING_PATH):
        print("label mapping is computed already")
        with open(LABEL_MAPPING_PATH, 'r') as f:
            labels_to_blocks = json.load(f)
        return labels_to_blocks
    block_shape = tentative_block_shape(scale, n_target_projects)
    n_blocks = make_subdivision_vol(scale, block_shape)
    labels_to_blocks = map_labels_to_blocks(n_blocks, tmp_folder, target, max_jobs)
    return labels_to_blocks


#
# make label division for subprojects
# by graph clustering
#

def compute_graph(scale, tmp_folder, target, max_jobs):
    task = GraphWorkflow
    config_dir = os.path.join(tmp_folder, 'configs')

    in_path = SEG_PATH
    in_key = os.path.join(SEG_KEY, 's%i' % scale)
    out_path = os.path.join(tmp_folder, 'data.n5')
    out_key = 'graph'

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=in_path, input_key=in_key,
             graph_path=out_path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Graph extraction failed"

    with z5py.File(out_path, 'r') as f:
        ds = f[out_key]
        edges = ds['edges'][:]
    n_nodes = int(edges.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)

    return graph


# we use the following features:
# nodes: the center of mass
# edges: the edge size
def compute_node_and_edge_features(uv_ids, scale, tmp_folder, target, max_jobs):
    task = EdgeFeaturesWorkflow
    config_dir = os.path.join(tmp_folder, 'configs')

    seg_path = SEG_PATH
    seg_key = os.path.join(SEG_KEY, 's%i' % scale)
    path = os.path.join(tmp_folder, 'data.n5')
    graph_key = 'graph'
    node_feat_key = 'node_features'
    edge_feat_key = 'features'

    in_path = BOUNDARY_PATH
    in_key = BOUNDARY_KEY + '/s%i' % scale

    with z5py.File(in_path, 'r') as f:
        shape = f[in_key].shape
    with z5py.File(seg_path, 'r') as f:
        assert f[seg_key].shape == shape

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=in_path, input_key=in_key,
             labels_path=seg_path, labels_key=seg_key,
             graph_path=path, graph_key=graph_key,
             output_path=path, output_key=edge_feat_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Edge feature computation failed"

    with z5py.File(path, 'r') as f:
        ds = f[edge_feat_key]
        edge_sizes = ds[:, -1]
    assert len(edge_sizes) == len(uv_ids)

    task = MorphologyWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=seg_path, input_key=seg_key,
             output_path=path, output_key=node_feat_key,
             prefix='feats')
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Node feature computation failed"

    with z5py.File(path, 'r') as f:
        ds = f[node_feat_key]
        node_sizes = ds[:, 1]
        node_pos = ds[:, 2:5]

    return node_pos, node_sizes, edge_sizes


def serialize_blocks(node_labels, tmp_folder, target, max_jobs):
    task = WriteLocal if target == 'local' else WriteSlurm
    config_dir = os.path.join(tmp_folder, 'configs')

    path = TMP_PATH
    seg_path = SEG_PATH
    seg_key = os.path.join(SEG_KEY, 's4')

    node_label_key = 'node_labels/clustering'
    out_key = 'volumes/clustering'
    with z5py.File(path) as f:
        ds = f.require_dataset(node_label_key, shape=node_labels.shape, chunks=node_labels.shape,
                               compression='gzip', dtype=node_labels.dtype)
        ds[:] = node_labels

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
             input_path=seg_path, input_key=seg_key,
             output_path=path, output_key=out_key,
             assignment_path=path, assignment_key=node_label_key,
             identifier='cluster-labels')
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Write failed"


def agglomerative_clustering_sklearn(graph, com, node_sizes, edge_sizes, n_target_projects):
    # uv ids to connectivity matrix
    # sparsify?
    print("Make connectivity matrix ...")
    uv_ids = graph.uvIds()
    n_nodes = graph.numberOfNodes
    conn_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    for u, v in uv_ids:

        # isolate the background label
        if u == 0 or v == 0:
            continue

        conn_matrix[u, v] = 1
        conn_matrix[v, u] = 1

    print("Run clustering ...")
    agglo = AgglomerativeClustering(n_clusters=n_target_projects, connectivity=conn_matrix)
    clustering = agglo.fit(com)
    node_labels = clustering.labels_
    assert len(node_labels) == len(com)
    return node_labels.astype('uint64')


def agglomerative_clustering_elf(graph, com,
                                 node_sizes, edge_sizes,
                                 n_target_projects, size_regularizer=1.):
    uv_ids = graph.uvIds()
    com_u = com[uv_ids[:, 0], :]
    com_v = com[uv_ids[:, 1], :]

    edge_features = np.sqrt(np.power(com_u - com_v, 2).sum(axis=1))
    assert len(edge_features) == len(edge_sizes)

    node_labels = agglomerative_clustering(graph, edge_features,
                                           node_sizes, edge_sizes,
                                           n_target_projects, size_regularizer=size_regularizer)
    return node_labels


def divide_labels_by_clustering(scale, n_target_projects,
                                tmp_folder, target, max_jobs):
    graph = compute_graph(scale, tmp_folder, target, max_jobs)
    uv_ids = graph.uvIds()
    com, node_sizes, edge_sizes = compute_node_and_edge_features(uv_ids, scale,
                                                                 tmp_folder, target, max_jobs)

    # node_labels = agglomerative_clustering_elf(graph, com, node_sizes, edge_sizes,
    #                                            n_target_projects, size_regularizer=1.)
    node_labels = agglomerative_clustering_sklearn(graph, com, node_sizes, edge_sizes, n_target_projects)
    vigra.analysis.relabelConsecutive(node_labels, start_label=1, keep_zeros=False, out=node_labels)
    n_projects = int(node_labels.max()) + 1
    node_labels[0] = 0

    labels_to_blocks = {ii: np.where(node_labels == ii)[0].tolist()
                        for ii in range(1, n_projects)}

    print("Number of segments per block:")
    for block_id, labels in labels_to_blocks.items():
        print("Block", block_id, ":", len(labels))
        if 0 in labels:
            print("Has bg label")

    with open(LABEL_MAPPING_PATH, 'w') as f:
        json.dump(labels_to_blocks, f)

    # serialize resulting segmentation for debugging
    serialize_blocks(node_labels, tmp_folder, target, max_jobs)
    return labels_to_blocks


#
# export segmentation and compute bounding boxes for sub projects
#


def make_root_seg(tmp_folder, target, max_jobs):
    in_path = SEG_PATH
    in_key = SEG_KEY + '/s0'
    ws_path = PAINTERA_PATH
    ws_key = PAINTERA_KEY + "/data/s0"
    out_path = TMP_PATH
    out_key = 'volumes/segmentation'
    assignment_out_key = 'node_labels/fragment_segment_assignment'

    config_dir = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_dir)
    tmp_path = os.path.join(tmp_folder, 'data.n5')

    # get the current fragment segment assignment
    assignments = node_labels(ws_path, ws_key,
                              in_path, in_key, 'rootseg',
                              tmp_folder, target=target, max_jobs=max_jobs,
                              max_overlap=True, ignore_label=None)

    # find the unique ids of the watersheds
    unique_key = 'uniques'
    find_uniques(ws_path, ws_key, tmp_path, unique_key,
                 tmp_folder, config_dir, max_jobs, target)

    with z5py.File(tmp_path, 'r') as f:
        ds = f[unique_key]
        ws_ids = ds[:]

    # convert to paintera fragment segment assignments
    id_offset = int(ws_ids.max()) + 1
    # print("Max ws id:", id_offset)
    # print("Ws  len  :", ws_ids.shape)
    # print("Ass len  :", assignments.shape)
    # print(ws_ids[-10:])
    assignments = assignments[ws_ids]
    assignments = vigra.analysis.relabelConsecutive(assignments,
                                                    start_label=id_offset,
                                                    keep_zeros=True)[0]
    assert len(assignments) == len(ws_ids), "%i, %i" % (len(assignments), len(ws_ids))
    paintera_assignments = np.concatenate([ws_ids[:, None], assignments[:, None]], axis=1).T

    assignment_tmp_key = 'tmp_assignments'
    with z5py.File(tmp_path) as f:
        ds = f.require_dataset(assignment_tmp_key, shape=paintera_assignments.shape,
                               compression='gzip', chunks=paintera_assignments.shape,
                               dtype='uint64')
        ds[:] = paintera_assignments

    # make and serialize new assignments
    print("Serializing assignments ...")
    serialize_assignments(tmp_folder,
                          tmp_path, assignment_tmp_key,
                          tmp_path, unique_key,
                          out_path, assignment_out_key,
                          locked_segments=None, relabel_output=False,
                          map_to_background=None)

    # write the new segmentation
    print("Serializing new segmentation ...")
    serialize_merged_segmentation(ws_path, ws_key,
                                  out_path, out_key,
                                  out_path, assignment_out_key,
                                  tmp_folder, max_jobs, target)


def compute_spatial_id_mapping(labels_to_blocks, tmp_folder, target, max_jobs):
    task = MorphologyWorkflow
    config_dir = os.path.join(tmp_folder, 'configs')

    scale = 0
    path = TMP_PATH
    path = SEG_PATH
    in_key = os.path.join(SEG_KEY, 's%i' % scale)

    out_key = 'morphology'

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key,
             output_path=path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret

    with z5py.File(path, 'r') as f:
        ds = f[out_key]
        morpho = ds[:]
    assert morpho.shape[1] == 11, "%i" % morpho.shape[1]

    rois_to_blocks = {}
    for block_id, labels in labels_to_blocks.items():
        if len(labels) == 0:
            rois_to_blocks[block_id] = None
            continue
        roi_start = morpho[labels, 5:8].astype('uint64').min(axis=0)
        roi_stop = morpho[labels, 8:11].astype('uint64').max(axis=0) + 1
        assert len(roi_start) == len(roi_stop) == 3
        rois_to_blocks[block_id] = (roi_start.tolist(),
                                    roi_stop.tolist())

    with open(ROI_PATH, 'w') as f:
        json.dump(rois_to_blocks, f)


def preprocess(n_target_projects, tmp_folder,
               use_graph_clustering=True, scale=2,
               target='slurm', max_jobs=200):

    if use_graph_clustering:
        labels_to_blocks = divide_labels_by_clustering(scale, n_target_projects,
                                                       tmp_folder, target, max_jobs)
    else:
        make_root_seg(tmp_folder, target, max_jobs)
        labels_to_blocks = divide_labels_by_blocking(scale, n_target_projects,
                                                     tmp_folder, target, max_jobs)

    # serialize the current paintera segmentation and map the ids to blocks spatially
    compute_spatial_id_mapping(labels_to_blocks, tmp_folder, target, max_jobs)


def copy_boundaries():
    from elf.parallel.operations import apply_operation_single

    n_threads = 48
    in_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    out_path = BOUNDARY_PATH

    def _copy_scale(scale):
        print("Copy volumes for scale", scale)

        in_key = 'volumes/affinities/s%i' % (scale + 1,)
        out_key = BOUNDARY_KEY + '/s%i' % scale

        f_in = z5py.File(in_path, 'r')
        ds_in = f_in[in_key]
        shape = ds_in.shape[1:]
        chunks = ds_in.chunks[1:]
        print("Have shape", shape)

        f_out = z5py.File(out_path)
        if out_key in f_out:
            print("Have copied scale", scale, "already")
            return

        ds_out = f_out.require_dataset(out_key, shape=shape, dtype=ds_in.dtype,
                                       chunks=chunks, compression='gzip')

        apply_operation_single(ds_in, np.mean, out=ds_out, axis=0,
                               n_threads=n_threads, verbose=True)

    for scale in range(0, 5):
        _copy_scale(scale)


if __name__ == '__main__':
    # copy_boundaries()

    n_projects = 50
    tmp_folder = './tmp_subdivision_labels'
    preprocess(n_projects, tmp_folder, target='local', max_jobs=48)
