import os
import json

import luigi
import numpy as np
import z5py

from cluster_tools.copy_volume import CopyVolumeLocal, CopyVolumeSlurm
from paintera_tools import convert_to_paintera_format, set_default_roi, set_default_block_shape
from mmpb.default_config import write_default_global_config
from common import RAW_PATH, RAW_KEY, PAINTERA_PATH, PAINTERA_KEY, TMP_PATH, ROI_PATH, LABEL_MAPPING_PATH


def write_max_id(path, key, max_id):
    with z5py.File(path) as f:
        ds = f[key]
        ds.attrs['maxId'] = max_id


def copy_watersheds(input_path, input_key,
                    output_path, output_key,
                    copy_ids, tmp_folder, target, max_jobs,
                    offset=None, insert_mode=False):
    task = CopyVolumeLocal if target == 'local' else CopyVolumeSlurm
    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    config = task.default_task_config()
    config.update({'insert_mode': insert_mode, 'offset': offset})

    if copy_ids is None:
        with z5py.File(PAINTERA_PATH, 'r') as f:
            max_id = f[PAINTERA_KEY].attrs['maxId']
    else:
        config.update({'value_list': copy_ids.tolist()})
        max_id = int(copy_ids.max())

    with open(os.path.join(config_dir, 'copy_volume.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs, config_dir=config_dir,
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             prefix='copy-ws')
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Copy failed"

    write_max_id(output_path, output_key, max_id)


def make_proofreading_project(project_folder, tmp_folder,
                              assignments, block_labels, block_roi,
                              target, max_jobs):

    if len(block_labels) == 0:
        return
    # don't do anything if we have a paintera project already
    if os.path.exists(os.path.join(project_folder, 'attributes.json')):
        return

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
    ds_ass = g_out.require_dataset('fragment-segment-assignment', shape=save_assignments.shape,
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
def make_proofreading_projects(root, labels_to_blocks, rois_to_blocks, target, max_jobs):
    os.makedirs(root, exist_ok=True)
    tmp_root = './tmps'
    os.makedirs(tmp_root, exist_ok=True)

    with z5py.File(TMP_PATH, 'r') as f:
        assignments = f['node_labels/fragment-segment-assignment2'][:]

    n_blocks = len(labels_to_blocks)
    # block_ids = range(1, n_blocks + 1)
    block_ids = range(3, 17)

    for block_id in block_ids:
        print("Make project", block_id, "/", n_blocks + 1)
        project_folder = os.path.join(root, 'project%02i' % block_id)
        tmp_folder = os.path.join(tmp_root, 'tmp_project%i' % block_id)
        make_proofreading_project(project_folder, tmp_folder,
                                  assignments, labels_to_blocks[block_id],
                                  rois_to_blocks[block_id], target, max_jobs)


def make_subdivision(root, target, max_jobs):
    with open(LABEL_MAPPING_PATH, 'r') as f:
        labels_to_blocks = json.load(f)
    labels_to_blocks = {int(k): v for k, v in labels_to_blocks.items()}

    with open(ROI_PATH, 'r') as f:
        rois_to_blocks = json.load(f)
    rois_to_blocks = {int(k): v for k, v in rois_to_blocks.items()}

    make_proofreading_projects(root, labels_to_blocks, rois_to_blocks, target, max_jobs)


if __name__ == '__main__':
    root = '/g/arendt/EM_6dpf_segmentation/corrections_and_proofreading/paintera_projects'
    target = 'local'
    max_jobs = 48
    make_subdivision(root, target, max_jobs)
