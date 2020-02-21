import os
import json
import luigi
from concurrent import futures

import nifty.tools as nt
from cluster_tools.postprocess import SizeFilterWorkflow
from cluster_tools.workflows import MulticutStitchingWorkflow

from elf.io import open_file
from ..cilia.segment import make_fg_mask, make_global_config, run_mws


def stitching_multicut(offsets, path, aff_key, seg_key, tmp_folder, target, max_jobs):
    task = MulticutStitchingWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    exp_path = os.path.join(tmp_folder, 'data.n5')
    assignment_key = 'node_labels/cuticle'

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
             labels_path=path, labels_key=seg_key,
             assignment_path=path, assignment_key=assignment_key,
             problem_path=exp_path, output_path=path, output_key=seg_key)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Multicut stitching failed")


def size_filter(path, seg_key, min_size, tmp_folder, target, max_jobs):
    task = SizeFilterWorkflow

    config_dir = os.path.join(tmp_folder, 'configs')

    # filter the segmentation objects by size
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=seg_key,
             output_path=path, output_key=seg_key,
             size_threshold=min_size)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Size filtering failed")


def map_to_foreground(path, seg_key, n_threads):
    with open_file(path, 'r') as f:
        ds = f[seg_key]
        shape = ds.shape
        block_shape = ds.chunks

        blocking = nt.blocking([0] * ds.ndim, shape, block_shape)

        def map_block(block_id):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            inp = ds[bb]
            if inp.sum() == 0:
                return
            inp = inp > 0
            ds[inp] = inp.astype(ds.dtype)

        with futures.TimeoutErrorhreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(map_block, block_id)
                     for block_id in range(blocking.numberOfBlocks)]
            [t.result() for t in tasks]


# NOTE this does not need to be so complicated. Probably we would get very similar results,
# if we thresholded the foreground prediction, then run ccs and filter out all small stuff.
# However, this is how we did it in the paper, so gonna leave it like this for now
def cuticle_segmentation_workflow(offsets, path,
                                  fg_key, aff_key, fg_mask_out_key, out_key,
                                  mask_path, mask_key,
                                  tmp_folder, target, max_jobs, n_threads):
    # preparation: find blocks we need to segment and write the global config
    with open_file(path, 'r') as f:
        shape = f[fg_key].shape
    block_shape = [64, 256, 256]
    make_global_config(mask_path, mask_key, shape, block_shape, tmp_folder, n_threads)

    # make foreground mask
    print("Make foreground mask ...")
    make_fg_mask(path, fg_key, fg_mask_out_key, tmp_folder, target, max_jobs)

    # segment blocks with mws
    print("Run mutex watershed ...")
    run_mws(offsets, path, fg_mask_out_key, aff_key, out_key,
            tmp_folder, target, max_jobs)

    # stitch block results with multicut
    print("Run multicut stitching ...")
    stitching_multicut(target, max_jobs)
    stitching_multicut(offsets, path, aff_key, out_key,
                       tmp_folder, target, max_jobs)

    print("Run size filter ...")
    min_size = 2000
    size_filter(path, out_key, min_size, tmp_folder, target, max_jobs)

    # map all remaining ids to foreground
    print("Mapping to foreground ...")
    map_to_foreground(path, out_key, n_threads)
