import os
import json
from concurrent import futures

import luigi
import numpy as np
import nifty.tools as nt
import vigra
import z5py
from cluster_tools.copy_volume import CopyVolumeLocal, CopyVolumeSlurm
from cluster_tools.morphology import MorphologyWorkflow
from paintera_tools import convert_to_paintera_format, set_default_roi, set_default_block_shape
from paintera_tools.util import find_uniques
from paintera_tools.serialize.serialize_from_commit import serialize_assignments, serialize_merged_segmentation

PAINTERA_PATH = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
PAINTERA_KEY = 'volumes/paintera/proofread_cells_multiset'

SEG_PATH = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/1.0.1/images',
                        'local/sbem-6dpf-1-whole-segmented-cells.n5')
SEG_KEY = 'setup0/timepoint0'

RAW_PATH = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5'
RAW_KEY = 'setup0/timepoint0'

TMP_PATH = './data.n5'


def get_blocking(scale, block_shape):
    g = z5py.File(PAINTERA_PATH)[PAINTERA_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    return shape, blocking


def tentative_block_shape(scale, n_target_blocks=50):
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


def make_root_seg(target, max_jobs):
    from mmpb.attributes.util import node_labels
    from mmpb.default_config import write_default_global_config

    in_path = SEG_PATH
    in_key = SEG_KEY + '/s0'
    ws_path = PAINTERA_PATH
    ws_key = PAINTERA_KEY + "/data/s0"
    out_path = TMP_PATH
    out_key = 'volumes/segmentation'
    assignment_out_key = 'node_labels/fragment_segment_assignment'

    tmp_folder = 'tmp_root_seg'
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


def map_labels_to_blocks(n_blocks, target, max_jobs):
    from mmpb.attributes.util import node_labels

    tmp_folder = './tmp_subdivision_labels'
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
    with open('./labels_to_blocks.json', 'w') as f:
        json.dump(labels_to_blocks, f)
    return labels_to_blocks


def write_max_id(path, key, copy_ids):
    with z5py.File(path) as f:
        ds = f[key]
        ds.attrs['maxId'] = int(copy_ids.max())


def copy_watersheds(input_path, input_key,
                    output_path, output_key,
                    copy_ids, tmp_folder, target, max_jobs):
    task = CopyVolumeLocal if target == 'local' else CopyVolumeSlurm
    config_dir = os.path.join(tmp_folder, 'configs')
    print(tmp_folder)

    config = task.default_task_config()
    config.update({'value_list': copy_ids.tolist()})
    with open(os.path.join(config_dir, 'copy_volume.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs, config_dir=config_dir,
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             prefix='copy-ws')
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Copy failed"

    write_max_id(output_path, output_key, copy_ids)


def make_proofreading_project(project_folder, tmp_folder,
                              assignments, block_labels, block_roi,
                              target, max_jobs):
    from mmpb.default_config import write_default_global_config

    os.makedirs(project_folder, exist_ok=True)
    config_dir = os.path.join(tmp_folder, 'configs')

    roi_begin, roi_end = block_roi
    write_default_global_config(config_dir, roi_begin, roi_end)
    with open(os.path.join(config_dir, 'global.config'), 'r') as f:
        block_shape = json.load(f)['block_shape']

    data_path = os.path.join(project_folder, 'data.n5')
    f = z5py.File(data_path)
    f.require_group('volumes')

    # make a link to the raw data
    raw_out_key = 'volumes/raw'
    if raw_out_key not in f:
        print("Make raw symlink")
        raw_in = os.path.join(RAW_PATH, RAW_KEY)
        raw_out = os.path.join(data_path, raw_out_key)
        os.symlink(raw_in, raw_out)

    # get the relevant fragment segment assignments for this block
    print("Get assignment mask")
    assignment_mask = np.isin(assignments[:, 1], block_labels)
    assert assignment_mask.sum() > 0
    block_assignments = assignments[assignment_mask]
    assert block_assignments.shape[0] == assignment_mask.sum()
    assert block_assignments.shape[1] == 2
    print("Sub assignments have the shape:", block_assignments.shape)

    # copy the relevant part of the fragment segment assignment
    print("Copy the assignments")
    g_out = f.require_group('volumes/paintera')
    save_assignments = block_assignments.T
    ds_ass = g_out.require_dataset('fragment_segment_assignment', shape=save_assignments.shape,
                                   chunks=save_assignments.shape, compression='gzip',
                                   dtype='uint64')
    ds_ass[:] = save_assignments

    # copy the relevant parts of the watersheds
    print("Copy the watersheds")
    ws_ids = block_assignments[:, 0]
    copy_watersheds(PAINTERA_PATH, os.path.join(PAINTERA_KEY, 'data/s0'),
                    data_path, 'volumes/watershed',
                    ws_ids, tmp_folder, target, max_jobs)

    # make the paintera data
    res = [0.025, 0.01, 0.01]
    restrict_sets = [-1, -1, 5, 4,
                     4, 3, 3, 1]
    print("Make new paintera data")
    set_default_roi(roi_begin, roi_end)
    set_default_block_shape(block_shape)
    convert_to_paintera_format(data_path, raw_out_key,
                               'volumes/watershed', 'volumes/paintera',
                               label_scale=1, resolution=res,
                               tmp_folder=tmp_folder, target=target, max_jobs=max_jobs,
                               max_threads=16, convert_to_label_multisets=True,
                               restrict_sets=restrict_sets)


# make the appropriate sub-volume and paintera project for each block
def make_proofreading_projects(labels_to_blocks, rois_to_blocks, target, max_jobs):
    root = '/g/arendt/EM_6dpf_segmentation/corrections_and_proofreading/paintera_projects'
    os.makedirs(root, exist_ok=True)
    tmp_root = './tmps'
    os.makedirs(tmp_root, exist_ok=True)

    with z5py.File(TMP_PATH, 'r') as f:
        assignments = f['node_labels/fragment_segment_assignment'][:]

    n_blocks = len(labels_to_blocks)
    block_ids = range(1, n_blocks + 1)
    block_ids = [8]

    for block_id in block_ids:
        print("Make project", block_id, "/", n_blocks + 1)
        project_folder = os.path.join(root, 'project%02i' % block_id)
        tmp_folder = os.path.join(tmp_root, 'tmp_project%i' % block_id)
        make_proofreading_project(project_folder, tmp_folder,
                                  assignments, labels_to_blocks[block_id],
                                  rois_to_blocks[block_id], target, max_jobs)


def compute_spatial_id_mapping(labels_to_blocks, target, max_jobs):
    task = MorphologyWorkflow
    tmp_folder = './tmp_subdivision_labels'
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

    with open('./rois_to_blocks.json', 'w') as f:
        json.dump(rois_to_blocks, f)


def preprocess():
    # map label ids to blocks
    scale = 2
    block_shape = tentative_block_shape(scale)
    n_blocks = make_subdivision_vol(scale, block_shape)
    target = 'slurm'
    max_jobs = 200
    make_root_seg(target, max_jobs)
    labels_to_blocks = map_labels_to_blocks(n_blocks, target, max_jobs)
    compute_spatial_id_mapping(labels_to_blocks, target, max_jobs)


def make_subdivision(target, max_jobs):
    with open('./labels_to_blocks.json', 'r') as f:
        labels_to_blocks = json.load(f)
    labels_to_blocks = {int(k): v for k, v in labels_to_blocks.items()}

    with open('./rois_to_blocks.json', 'r') as f:
        rois_to_blocks = json.load(f)
    rois_to_blocks = {int(k): v for k, v in rois_to_blocks.items()}

    make_proofreading_projects(labels_to_blocks, rois_to_blocks, target, max_jobs)


if __name__ == '__main__':
    # preprocess()

    # fixup()

    target = 'local'
    max_jobs = 48
    make_subdivision(target, max_jobs)
