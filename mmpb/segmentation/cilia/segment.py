import os
import json
import luigi
import numpy as np

from cluster_tools.mutex_watershed import MwsWorkflow
from cluster_tools.postprocess import SizeFilterWorkflow
from cluster_tools.thresholded_components.threshold import ThresholdLocal, ThresholdSlurm
from cluster_tools.workflows import MulticutStitchingWorkflow

import elf.parallel
from elf.io import open_file
from z5py.util import copy_dataset

from mmpb.default_config import get_default_shebang
from mmpb.segmentation.network.prediction import prefilter_blocks


def make_global_config(mask_path, mask_key, shape, block_shape, tmp_folder, n_threads):
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    config = MwsWorkflow.get_config()['global']
    shebang = get_default_shebang()
    block_list_path = os.path.join(tmp_folder, 'blocks.json')
    prefilter_blocks(mask_path, mask_key,
                     shape, block_shape,
                     block_list_path, n_threads=n_threads)
    config.update({'shebang': shebang,
                   'block_shape': block_shape,
                   'block_list_path': block_list_path})

    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(config, f)


def run_mws(offsets, path, fg_key, aff_key, out_key,
            tmp_folder, target, max_jobs,
            strides=[6, 6, 6], halo=[16, 32, 32]):

    config_folder = os.path.join(tmp_folder, 'configs')
    configs = MwsWorkflow.get_config()

    config = configs['mws_blocks']
    config.update({'strides': strides, 'time_limit': 360, 'mem_limit': 12,
                   'randomize_strides': True, 'noise_level': 1e-4})
    with open(os.path.join(config_folder, 'mws_blocks.config'), 'w') as f:
        json.dump(config, f)

    task = MwsWorkflow(tmp_folder=tmp_folder, max_jobs=max_jobs,
                       target=target, config_dir=config_folder,
                       input_path=path, input_key=aff_key,
                       output_path=path, output_key=out_key,
                       mask_path=path, mask_key=fg_key,
                       offsets=offsets, halo=halo)
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("MWS failed")


def make_fg_mask(path, fg_key, fg_mask_out_key, tmp_folder, target, max_jobs):
    task = ThresholdLocal if target == 'local' else ThresholdSlurm

    threshold = .5
    config_folder = os.path.join(tmp_folder, 'configs')
    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=max_jobs,
             input_path=path, input_key=fg_key,
             output_path=path, output_key=fg_mask_out_key,
             threshold=threshold, threshold_mode='greater')
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Threshold failed")


def find_bounding_boxes(seg_path, seg_key, n_threads, scale_factor):
    with open_file(seg_path, 'r') as f:
        ds = f[seg_key]
        ds.n_threads = n_threads
        chunks = ds.chunks
        seg = ds[:]

    unique_segs = elf.parallel.unique(seg, block_shape=chunks,
                                      n_threads=n_threads, verbose=True)[1:]

    bbs = []
    # could use more efficient impl from scipy/skimage
    for seg_id in unique_segs:
        where_seg = np.where(seg == seg_id)
        bb = tuple(slice(int(ws.min()) * sf,
                         (int(ws.max()) + 1) * sf) for ws, sf in zip(where_seg, scale_factor))
        bbs.append(bb)

    return bbs


def set_bounding_box(tmp_folder, bb):
    config_path = os.path.join(tmp_folder, 'configs', 'global.config')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # clear block list path
    config['block_list_path'] = None

    # set the bounding box
    bb_start = [b.start for b in bb]
    bb_stop = [b.stop for b in bb]
    config.update({'roi_begin': bb_start, 'roi_end': bb_stop})
    with open(config_path, 'w') as f:
        json.dump(config, f)


def stitching_multicut(offsets, path, aff_key, ws_key,
                       tmp_folder, config_folder, target, max_jobs):
    task = MulticutStitchingWorkflow

    exp_path = os.path.join(tmp_folder, 'data.n5')
    assignment_key = 'node_labels/cilia'
    out_key = 'volumes/stitched_cilia'

    configs = task.get_config()
    config = configs['probs_to_costs']
    config.update({'weight_edges': True})
    with open(os.path.join(config_folder, 'probs_to_costs.config'), 'w') as f:
        json.dump(config, f)

    config = configs['block_edge_features']
    config.update({'offsets': offsets[:6]})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(config, f)

    task_names = ['merge_sub_graphs', 'merge_edge_features']
    for tname in task_names:
        config = configs[tname]
        config.update({'time_limit': 120, 'mem_limit': 32})
        with open(os.path.join(config_folder, '%s.config' % tname), 'w') as f:
            json.dump(config, f)

    config = configs['stitching_multicut']
    config.update({'time_limit': 600, 'mem_limit': 128, 'threads_per_job': 8})
    with open(os.path.join(config_folder, 'stitching_multicut.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=aff_key,
             labels_path=path, labels_key=ws_key,
             assignment_path=exp_path, assignment_key=assignment_key,
             problem_path=exp_path,
             output_path=exp_path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Multicut stitching failed")


def postprocess_and_write(path, out_key, bb, tmp_folder, config_dir,
                          target, max_jobs, n_threads, offset, min_size):
    task = SizeFilterWorkflow

    exp_path = os.path.join(tmp_folder, 'data.n5')
    seg_in_key = 'volumes/stitched_cilia'
    seg_out_key = 'volumes/filtered_cilia'

    # filter the segmentation objects by size
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=exp_path, input_key=seg_in_key,
             output_path=exp_path, output_key=seg_out_key,
             size_threshold=min_size)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Size filtering failed")

    with open_file(exp_path) as fin, open_file(path) as fout:
        ds_in, ds_out = fin[seg_out_key], fout[out_key]

        # apply offset to the segmentation
        elf.parallel.add(ds_in, offset, out=ds_in, mask=ds_in,
                         n_threads=n_threads, roi=bb)

        # copy to the output
        copy_dataset(exp_path, path, seg_out_key, out_key,
                     n_threads=n_threads, roi=bb)

        # find new offset
        offset = elf.parallel.max(ds_out, n_threads=n_threads, roi=bb)

    return offset


def get_scale_factor(path, key, mask_path, mask_key):
    with open_file(path, 'r') as f, open_file(mask_path, 'r') as fm:
        shape = f[key].shape
        mask_shape = fm[mask_key].shape
    scale_factor = [int(round(sh / float(ms), 0)) for sh, ms in zip(shape, mask_shape)]
    return scale_factor


def cilia_segmentation_workflow(offsets, path,
                                fg_key, aff_key, fg_mask_out_key, out_key,
                                mask_path, mask_key,
                                tmp_folder, target, max_jobs, n_threads):

    size_threshold = 1000
    # preparation: find blocks we need to segment and write the global config
    with open_file(path, 'r') as f:
        shape = f[fg_key].shape
    block_shape = [64, 256, 256]
    make_global_config(mask_path, mask_key, shape, block_shape, tmp_folder, n_threads)

    # make mask for the foreground
    print("Make foreground mask ...")
    make_fg_mask(path, fg_key, fg_mask_out_key, tmp_folder, target, max_jobs)

    # run block-wise mws
    print("Run mutex watershed ...")
    run_mws(offsets, path, fg_mask_out_key, aff_key, out_key,
            tmp_folder, target, max_jobs)

    # find bounding box(es) of current segments mask and set it
    scale_factor = get_scale_factor(path, fg_key, mask_path, mask_key)
    bbs = find_bounding_boxes(mask_path, mask_key, n_threads, scale_factor)

    offset = 0
    # run multicut for the bounding boxes
    config_folder = os.path.join(tmp_folder, 'configs')
    print("Run stitching multicuts ...")
    for ii, bb in enumerate(bbs):
        print("for bounding box", ii, ":", bb)
        set_bounding_box(tmp_folder, bb)
        tmp_folder_mc = os.path.join(tmp_folder, 'tmps_mc', 'tmp_%i' % ii)
        os.makedirs(tmp_folder_mc, exist_ok=True)
        stitching_multicut(offsets, path, aff_key, out_key,
                           tmp_folder, config_folder, target, max_jobs)

        # run filters and update offset
        offset = postprocess_and_write(path, out_key, bb,
                                       tmp_folder_mc, config_folder,
                                       target, max_jobs, n_threads,
                                       offset, size_threshold)
