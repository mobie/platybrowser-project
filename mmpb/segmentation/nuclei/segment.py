import os
import json
import luigi
from cluster_tools.mutex_watershed import MwsWorkflow
from cluster_tools.postprocess import SizeFilterWorkflow
from elf.io import open_file
from elf.parallel import greater_equal
from elf.wrapper import NormalizeWrapper
from elf.wrapper.resized_volume import ResizedVolume
from mmpb.default_config import get_default_shebang

STITCH_MODES = {'biased', 'unbiased', ''}


def get_chunks(chunks, path, key):
    if chunks is None:
        with open_file(path, 'r') as f:
            chunks = f[key].chunks
    return chunks


def make_foreground_mask(path, input_key, output_key,
                         mask_path, mask_key,
                         threshold, chunks, n_threads):

    with open_file(path, 'a') as f, open_file(mask_path, 'r') as f_mask:
        ds = NormalizeWrapper(f[input_key])
        ds_out = f.require_dataset(output_key, shape=ds.shape, compression='gzip',
                                   dtype='uint8', chunks=chunks)

        ds_mask = f_mask[mask_key]
        ds_mask = ResizedVolume(ds_mask, shape=ds.shape, order=0)

        greater_equal(ds, threshold, out=ds_out, verbose=True,
                      n_threads=n_threads, mask=ds_mask)


def mws_segmentation(offsets, path, input_key, fg_mask_key, output_key,
                     tmp_folder, target, max_jobs, stitch_mode):
    task = MwsWorkflow
    qos = 'normal'

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    configs = task.get_config()

    # we use a smaller block shape to speed up MWS
    block_shape = [64, 256, 256]
    conf = configs['global']
    shebang = get_default_shebang()
    block_shape = block_shape
    conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(conf, f)

    # write config for mws block task
    strides = [4, 4, 4]
    conf = configs['mws_blocks']
    conf.update({'randomize_strides': True, 'strides': strides, 'mem_limit': 12, 'time_limit': 900})
    with open(os.path.join(config_folder, 'mws_blocks.config'), 'w') as f:
        json.dump(conf,  f)

    # determine config for the given stitching mode
    if stitch_mode == '':
        stitch_mc = False
    elif stitch_mode == 'biased':
        stitch_mc = True
        beta1, beta2 = 0.5, 0.75
    elif stitch_mode == 'unbiased':
        stitch_mc = True
        beta1 = beta2 = 0.5

    if stitch_mc:
        # write config for stitching multicut
        conf = configs['stitching_multicut']
        conf.update({'beta1': beta1, 'beta2': beta2, 'qos': qos})
        with open(os.path.join(config_folder, 'stitching_multicut.config'), 'w') as f:
            json.dump(conf, f)

        conf = configs['write']
        conf.update({'mem_limit': 8, 'time_limit': 120, 'qos': qos})
        with open(os.path.join(config_folder, 'write.config'), 'w') as f:
            json.dump(conf, f)

        # write config for edge feature task
        conf = configs['block_edge_features']
        conf.update({'offsets': offsets, 'mem_limit': 4, 'qos': qos})
        with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
            json.dump(conf, f)

        conf_names = ['merge_edge_features', 'merge_sub_graphs',
                      'map_edge_ids', 'simple_stitch_assignments']
        for name in conf_names:
            conf = configs[name]
            conf.update({'mem_limit': 128, 'time_limit': 240, 'threads_per_job': 16, 'qos': qos})
            with open(os.path.join(config_folder, '%s.config' % name), 'w') as f:
                json.dump(conf, f)

        conf = configs['stitching_multicut']
        # set time limit for the multicut task to 18 hours (in minutes)
        tlim_task = 18 * 60
        # set time limit for the solver to 16 hours (in seconds)
        tlim_solver = 16 * 60 * 60
        conf.update({'mem_limit': 256, 'time_limit': tlim_task, 'threads_per_job': 16, 'qos': qos,
                     'agglomerator': 'greedy-additive', 'time_limit_solver': tlim_solver})
        with open(os.path.join(config_folder, 'stitching_multicut.config'), 'w') as f:
            json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=input_key,
             output_path=path, output_key=output_key,
             mask_path=path, mask_key=fg_mask_key,
             stitch_via_mc=stitch_mc, offsets=offsets)

    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Mws segmentation failed")


def run_size_filter(path, key, tmp_folder, target, max_jobs):

    # size threshold from histograms
    # size_threshold = int(18779.890625)
    # approximate size filter
    size_threshold = 15000

    config_folder = os.path.join(tmp_folder, 'configs')
    task = SizeFilterWorkflow(tmp_folder=tmp_folder, max_jobs=max_jobs,
                              target=target, config_dir=config_folder,
                              input_path=path, input_key=key,
                              output_path=path, output_key=key,
                              size_threshold=size_threshold, relabel=True)
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("Filter sizes failed")


def nucleus_segmentation_workflow(offsets, path, forgeround_key, affinity_key,
                                  mask_path, mask_key,
                                  seg_out_key, mask_out_key,
                                  tmp_folder, target, max_jobs, n_threads,
                                  threshold=.5, stitch_mode='biased', chunks=None):
    assert stitch_mode in STITCH_MODES, "Invalid stitching mode %s, choose one of %s" % (stitch_mode,
                                                                                         str(STITCH_MODES))
    chunks = get_chunks(chunks, path, forgeround_key)
    print("Make foreground mask ...")
    make_foreground_mask(path, forgeround_key, mask_out_key,
                         mask_path, mask_key, threshold,
                         chunks, n_threads)

    print("Segment nuclei with mws ...")
    mws_segmentation(offsets, path, affinity_key, mask_out_key, seg_out_key,
                     tmp_folder, target, max_jobs, stitch_mode)

    print("Size filter the nucleus segmentation ...")
    run_size_filter(path, seg_out_key, tmp_folder, target, max_jobs)
