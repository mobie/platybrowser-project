import os
import luigi
import numpy as np
from concurrent import futures
from tqdm import tqdm

import nifty.tools as nt
import vigra
from cluster_tools.postprocess import SizeFilterWorkflow
from cluster_tools.utils.volume_utils import normalize

from elf.io import open_file
from ..cilia.segment import make_fg_mask, make_global_config, run_mws


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


def filter_background(path, fg_key, seg_key, block_shape, n_threads):
    with open_file(path) as f:
        ds_seg = f[seg_key]
        ds_fg = f[fg_key]
        shape = ds_seg.shape

        blocking = nt.blocking([0] * ds_seg.ndim, shape, block_shape)

        def filter_block(block_id):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            seg = ds_seg[bb].astype('uint32')
            if seg.sum() == 0:
                return

            inp = normalize(ds_fg[bb])
            mean_fg = vigra.analysis.extractRegionFeatures(inp, seg, features=['mean'])['mean']
            fg_ids = np.where(mean_fg > .5)[0]
            filtered = np.isin(seg, fg_ids)
            ds_seg[bb] = filtered.astype(ds_seg.dtype)

        n_blocks = blocking.numberOfBlocks
        with futures.ThreadPoolExecutor(n_threads) as tp:
            list(tqdm(tp.map(filter_block, range(n_blocks)), total=n_blocks))


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

    # FIXME results look bad form here on, investigate
    print("Run size filter ...")
    min_size = 500
    size_filter(path, out_key, min_size, tmp_folder, target, max_jobs)

    # map all remaining ids to foreground
    print("Filter background ...")
    filter_background(path, fg_key, out_key, block_shape, n_threads)
