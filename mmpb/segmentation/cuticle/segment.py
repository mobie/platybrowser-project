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
    tmp_size_filter = os.path.join(tmp_folder, 'size_filtering')

    # filter the segmentation objects by size
    t = task(tmp_folder=tmp_size_filter, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=seg_key,
             output_path=path, output_key=seg_key,
             size_threshold=min_size)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Size filtering failed")


class FilterTask(luigi.Task):
    path = luigi.Parameter()
    fg_key = luigi.Parameter()
    seg_key = luigi.Parameter()
    block_shape = luigi.ListParameter()
    n_threads = luigi.IntParameter()
    out_path = luigi.Parameter()
    threshold = luigi.FloatParameter()

    def run(self):
        with open_file(self.path) as f:
            ds_seg = f[self.seg_key]
            ds_fg = f[self.fg_key]
            shape = ds_seg.shape

            blocking = nt.blocking([0] * ds_seg.ndim, shape, self.block_shape)

            def filter_block(block_id):
                block = blocking.getBlock(block_id)
                bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                seg = ds_seg[bb].astype('uint32')
                if seg.sum() == 0:
                    return

                inp = normalize(ds_fg[bb])
                mean_fg = vigra.analysis.extractRegionFeatures(inp, seg, features=['mean'])['mean']
                fg_ids = np.where(mean_fg > self.threshold)[0]
                filtered = np.isin(seg, fg_ids)
                ds_seg[bb] = filtered.astype(ds_seg.dtype)

            n_blocks = blocking.numberOfBlocks
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                list(tqdm(tp.map(filter_block, range(n_blocks)), total=n_blocks))

        # dummy output for luigi
        with open(self.out_path, 'w') as f:
            f.write("Success!")

    def output(self):
        return luigi.LocalTarget(self.out_path)


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

    print("Run size filter ...")
    min_size = 1000
    size_filter(path, out_key, min_size, tmp_folder, target, max_jobs)

    # map all remaining ids to foreground
    print("Filter background ...")
    threshold = .75
    task = FilterTask(path=path, fg_key=fg_key, seg_key=out_key,
                      block_shape=block_shape, n_threads=n_threads,
                      threshold=threshold,
                      out_path=os.path.join(tmp_folder, 'filter_task.log'))
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("Background filtering failed")
