import os
import json
import luigi
from concurrent import futures

import numpy as np
import nifty.tools as nt
import vigra
import z5py
from cluster_tools.morphology import MorphologyWorkflow
from mmpb.attributes.util import node_labels
from paintera_tools.serialize.serialize_from_commit import serialize_assignments, serialize_merged_segmentation
from paintera_tools.util import find_uniques

from common import (PAINTERA_PATH, PAINTERA_KEY,
                    SEG_PATH, SEG_KEY, TMP_PATH,
                    LABEL_MAPPING_PATH, ROIS_PATH)


#
# make label division for subprojects
#


def get_blocking(scale, block_shape):
    g = z5py.File(PAINTERA_PATH)[PAINTERA_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
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
# export segmentation and compute bounding boxes for sub projects
#


def make_root_seg(tmp_folder, target, max_jobs):
    from mmpb.attributes.util import node_labels
    from mmpb.default_config import write_default_global_config

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

    path = TMP_PATH
    in_key = 'volumes/segmentation'
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

    with open(ROIS_PATH, 'w') as f:
        json.dump(rois_to_blocks, f)


def preprocess(n_target_projects, tmp_folder,
               use_graph_clustering=False,
               target='slurm', max_jobs=200):

    # TODO it would be cleaner to get the mapping of ids to
    # target blocks by some means of graph clustering instead of mapping to blocks
    if use_graph_clustering:
        raise NotImplementedError("TODO")
    else:
        scale = 2
        labels_to_blocks = divide_labels_by_blocking(scale, n_target_projects,
                                                     tmp_folder, target, max_jobs)

    # serialize the current paintera segmentation and map the ids to blocks spatially
    make_root_seg(tmp_folder, target, max_jobs)
    compute_spatial_id_mapping(labels_to_blocks, tmp_folder, target, max_jobs)


if __name__ == '__main__':
    n_projects = 50
    tmp_folder = './tmp_subdivision_labels'
    preprocess(n_projects, tmp_folder)
