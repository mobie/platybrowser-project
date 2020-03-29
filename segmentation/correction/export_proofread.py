import os
import json
import luigi
import numpy as np
import z5py

from cluster_tools.downscaling import DownscalingWorkflow
from paintera_tools.serialize.serialize_from_commit import (serialize_assignments,
                                                            serialize_merged_segmentation)
from paintera_tools.util import find_uniques
from mmpb.default_config import write_default_global_config, set_default_block_shape
from mmpb.util import add_max_id
from common import PAINTERA_PATH, PAINTERA_KEY, TMP_PATH, ROI_PATH
from make_proofreading_projects import copy_watersheds

PROJECT_ROOT = '/g/arendt/EM_6dpf_segmentation/corrections_and_proofreading/paintera_projects'


def downscale(path, in_key, out_key, tmp_folder, max_jobs, target, n_scales=5):
    task = DownscalingWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_folder)
    configs = task.get_config()

    config = configs['downscaling']
    config.update({'mem_limit': 8, 'time_limit': 120,
                   'library_kwargs': {'order': 0}})
    with open(os.path.join(config_folder, 'downscaling.config'), 'w') as f:
        json.dump(config, f)

    scale_factors = [[2, 2, 2]] * n_scales
    halos = [[0, 0, 0]] * n_scales

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=in_key, output_key_prefix=out_key,
             scale_factors=scale_factors, halos=halos,
             metadata_format='paintera')
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Downscaling the segmentation failed")

    with z5py.File(path, 'r') as f:
        ds = f[in_key]
        max_id = ds.attrs['maxId']

    for scale in range(n_scales + 1):
        scale_key = '%s/s%i' % (out_key, scale)
        add_max_id(path, scale_key, max_id=max_id)


def serialize_segmentation(ws_path, ws_key, out_path, out_key, assignments,
                           tmp_folder, target, max_jobs):

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    config_dir = os.path.join(tmp_folder, 'configs')
    # find the unique ids of the watersheds
    unique_key = 'uniques'
    find_uniques(ws_path, ws_key, tmp_path, unique_key,
                 tmp_folder, config_dir, max_jobs, target)

    with z5py.File(tmp_path, 'r') as f:
        ds = f[unique_key]
        ws_ids = ds[:]

    ws_assignments = dict(zip(assignments[:, 0], assignments[:, 1]))
    assignments = np.array([ws_assignments.get(ws_id, ws_id) for ws_id in ws_ids])

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
    assignment_out_key = 'node_labels/assignments2'
    serialize_assignments(tmp_folder,
                          tmp_path, assignment_tmp_key,
                          tmp_path, unique_key,
                          out_path, assignment_out_key,
                          locked_segments=None, relabel_output=True,
                          map_to_background=None)

    # write the new segmentation
    print("Serializing new segmentation ...")
    serialize_merged_segmentation(ws_path, ws_key,
                                  out_path, out_key,
                                  out_path, assignment_out_key,
                                  tmp_folder, max_jobs, target)

    out_key_prefix = os.path.split(out_key)[0]
    downscale(out_path, out_key, out_key_prefix, tmp_folder, max_jobs, target)


def export_selected_projects(projects, rois_to_blocks, target, max_jobs):
    """ Export only selected projects and fill in the rest with the
    old global paintera project. This means we need to keep ids consistent
    between projects.
    """
    project_folders = [os.path.join(PROJECT_ROOT, 'project%02i' % project_id)
                       for project_id in projects]
    assert all(os.path.exists(pfolder) for pfolder in project_folders)

    tmp_folder = './tmp_export'
    tmp_path = os.path.join(tmp_folder, 'data.n5')

    #
    # load the original paintera data
    #

    # copy the watershed segmentation
    ws_in_key = os.path.join(PAINTERA_KEY, 'data', 's0')
    ws_out_key = 'volumes/watershed'
    copy_watersheds(PAINTERA_PATH, ws_in_key,
                    tmp_path, ws_out_key,
                    None, tmp_folder, target, max_jobs)
    with z5py.File(tmp_path, 'r') as f:
        max_id = f[ws_out_key].attrs['maxId']

    # load the fragment segment assignments
    ass_key = os.path.join(PAINTERA_KEY, 'fragment-segment-assignment')
    with z5py.File(PAINTERA_PATH, 'r') as f:
        assignments = f[ass_key][:].T

    #
    # load corrections from the projects and insert them
    #

    for project_folder in project_folders:
        proj_id = int(project_folder[-2:])
        tmp_project = os.path.join(tmp_folder, 'tmp_proj%i' % proj_id)
        project_path = os.path.join(project_folder, 'data.n5')
        project_in_root = 'volumes/paintera'
        project_in_key = os.path.join(project_in_root, 'data', 's0')

        # set the bounding box for this project
        config_dir = os.path.join(tmp_project, 'configs')
        rb, re = rois_to_blocks[proj_id]
        set_default_block_shape([50, 512, 512])
        write_default_global_config(config_dir, rb, re)

        # copy this watersheds, offsetting everything with the current max id
        copy_watersheds(project_path, project_in_key,
                        tmp_path, ws_out_key,
                        None, tmp_project, target, max_jobs,
                        offset=max_id, insert_mode=True)

        # update the fragment segment assignment
        project_ass_key = os.path.join(project_in_root, 'fragment-segment-assignment')
        with z5py.File(project_path, 'r') as f:
            this_assignments = f[project_ass_key][:].T
        # offset the assignments
        this_assignments += max_id
        assignments = np.concatenate([assignments, this_assignments], axis=0)

        # update the max id
        max_id = int(assignments.max())

    # write the new segmentation
    seg_out_key = 'volumes/segmentation2/s0'
    serialize_segmentation(tmp_path, ws_out_key, TMP_PATH, seg_out_key, assignments,
                           tmp_folder, target, max_jobs)


def export_all_projects():
    """ Export from all proof-reading projects.
    This means we don't need to retrieve any data from the original paintera project.
    """


def check_exported(scale=3):
    from heimdall import view, to_source
    path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5'
    key = 'setup0/timepoint0/s%i' % (scale + 1,)

    f = z5py.File(path, 'r')
    ds = f[key]
    ds.n_threads = 8
    print(ds.shape)
    raw = ds[:]

    path = './data.n5'
    key = 'volumes/segmentation2/s%i' % scale
    f = z5py.File(path, 'r')
    ds = f[key]
    print(ds.shape)
    ds.n_threads = 8
    seg = ds[:].astype('uint32')

    view(to_source(raw, name='raw'),
         to_source(seg, name='segmentation'))


def first_export():
    with open(ROI_PATH, 'r') as f:
        rois_to_blocks = json.load(f)
    rois_to_blocks = {int(k): v for k, v in rois_to_blocks.items()}

    target = 'local'
    max_jobs = 48
    export_selected_projects([8], rois_to_blocks, target, max_jobs)


if __name__ == '__main__':
    check_exported()
