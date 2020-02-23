import os
import json
import luigi

from cluster_tools import MulticutSegmentationWorkflow, LiftedMulticutSegmentationWorkflow
from cluster_tools.watershed import WatershedWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow

from mmpb.default_config import write_default_global_config
from mmpb.extension.segmentation.nucleus_assignments import NucleusAssignmentWorkflow
from mmpb.extension.segmentation.unmerge import UnmergeWorkflow


def run_watershed(path, aff_path, use_curated_affs,
                  mask_path, mask_key,
                  tmp_folder, target, max_jobs):
    config_folder = os.path.join(tmp_folder, 'configs')

    configs = WatershedWorkflow.get_config()
    ws_config = configs['watershed']
    ws_config.update({'threshold': .25, 'alpha': 0.9,
                      'sigma_weights': 2.0, 'sigma_seeds': 2.0,
                      'apply_dt_2d': False, 'apply_ws_2d': False,
                      'halo': [4, 32, 32], 'size_filter': 100,
                      'channel_begin': 0, 'channel_end': 3,
                      'non_maximum_suppression': False,
                      'time_limit': 360, 'mem_limit': 20})
    with open(os.path.join(config_folder, 'watershed.config'), 'w') as f:
        json.dump(ws_config, f)

    write_config = configs['write']
    write_config.update({'mem_limit': 16})
    with open(os.path.join(config_folder, 'write.config'), 'w') as f:
        json.dump(write_config, f)

    if use_curated_affs:
        aff_key = 'volumes/cells/curated_affinities/s1'
        output_key = 'volumes/cells/curated_watershed'
        tmp_ws = os.path.join(tmp_folder, 'tmp_curated_ws')
    else:
        aff_key = 'volumes/cells/affinities/s1'
        output_key = 'volumes/cells/watershed'
        tmp_ws = os.path.join(tmp_folder, 'tmp_ws')

    task = WatershedWorkflow(tmp_folder=tmp_ws, config_dir=config_folder,
                             max_jobs=max_jobs, target=target,
                             input_path=aff_path, input_key=aff_key,
                             mask_path=path, mask_key=mask_key,
                             output_path=path, output_key=output_key,
                             two_pass=False)
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("Watershed failed")


def run_mc(path, aff_path, use_curated_affs,
           tmp_folder, target,
           max_threads, max_jobs, max_jobs_mc):
    task = MulticutSegmentationWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    if use_curated_affs:
        input_key = 'volumes/cells/curated_affinities/s1'
        ws_key = 'volumes/cells/curated_watershed'
        assignment_key = 'node_labels/cells/curated_mc/result'
        out_key = 'volumes/cells/curated_mc/result'
        tmp_mc = os.path.join(tmp_folder, 'tmp_mc_curated')
    else:
        input_key = 'volumes/cells/affinities/s1'
        ws_key = 'volumes/cells/watershed'
        assignment_key = 'node_labels/cells/mc/result'
        out_key = 'volumes/cells/mc/result'
        tmp_mc = os.path.join(tmp_folder, 'tmp_mc')
    exp_path = os.path.join(tmp_mc, 'problem.n5')

    configs = task.get_config()
    subprob_config = configs['solve_subproblems']
    subprob_config.update({'threads_per_job': max_threads,
                           'time_limit': 720,
                           'mem_limit': 64,
                           'time_limit_solver': 60*60*6})
    with open(os.path.join(config_folder, 'solve_subproblems.config'), 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(feat_config, f)

    exponent = 1.
    weight_edges = True
    costs_config = configs['probs_to_costs']
    costs_config.update({'weight_edges': weight_edges,
                         'weighting_exponent': exponent,
                         'mem_limit': 16})
    with open(os.path.join(config_folder, 'probs_to_costs.config'), 'w') as f:
        json.dump(costs_config, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'map_edge_ids',
             'reduce_problem', 'solve_global']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads if tt != 'reduce_problem' else 8,
                       'mem_limit': 128,
                       'time_limit': 1440,
                       'qos': 'normal',
                       'agglomerator': 'decomposition-gaec',
                       'time_limit_solver': 60*60*15})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    t = task(tmp_folder=tmp_mc, config_dir=config_folder, target=target,
             max_jobs=max_jobs, max_jobs_multicut=max_jobs_mc,
             input_path=path, input_key=input_key,
             ws_path=path, ws_key=ws_key,
             problem_path=exp_path,
             node_labels_key=assignment_key,
             output_path=path, output_key=out_key,
             n_scales=1, sanity_checks=False, skip_ws=True)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Multicut failed")


def run_lmc(path, aff_path, use_curated_affs,
            region_path, region_key,
            tmp_folder, target,
            max_threads, max_jobs, max_jobs_mc):
    task = LiftedMulticutSegmentationWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')
    nucleus_seg_key = 'volumes/nuclei/segmentation'

    if use_curated_affs:
        input_key = 'volumes/cells/curated_affinities/s1'
        ws_key = 'volumes/cells/curated_watershed'
        assignment_key = 'node_labels/curated_lmc/result'
        out_key = 'volumes/cells/curated_lmc/result'
        tmp_lmc = os.path.join(tmp_folder, 'tmp_lmc_curated')

        clear_path, clear_key = region_path, region_key
        node_label_dict = {'ignore_transition': (path, clear_key)}
    else:
        input_key = 'volumes/cells/affinities/s1'
        ws_key = 'volumes/cells/watershed'
        assignment_key = 'node_labels/lmc/result'
        out_key = 'volumes/cells/lmc/result'
        tmp_lmc = os.path.join(tmp_folder, 'tmp_lmc')
        clear_path = clear_key = node_label_dict = None

    # exp path
    exp_path = os.path.join(tmp_lmc, 'problem.n5')

    configs = task.get_config()
    subprob_config = configs['solve_lifted_subproblems']
    subprob_config.update({'threads_per_job': max_threads,
                           'time_limit': 1200,
                           'mem_limit': 384,
                           'time_limit_solver': 60*60*4})
    with open(os.path.join(config_folder,
                           'solve_lifted_subproblems.config'), 'w') as f:
        json.dump(subprob_config, f)

    tasks = ['reduce_lifted_problem', 'solve_lifted_global', 'clear_lifted_edges_from_labels']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads,
                       'mem_limit': 256,
                       'time_limit': 2160,
                       # 'agglomerator': 'greedy-additive',
                       'agglomerator': 'kernighan-lin',
                       'time_limit_solver': 60*60*15})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    tasks = ['block_node_labels', 'merge_node_labels']
    for tt in tasks:
        config = configs[tt]
        config.update({"time_limit": 60, "mem_limit": 16})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    conf = configs['sparse_lifted_neighborhood']
    conf.update({'time_limit': 180, 'mem_limit': 256, 'threads_per_job': max_threads})
    with open(os.path.join(config_folder, 'sparse_lifted_neighborhood.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_lmc, config_dir=config_folder, target=target,
             max_jobs=max_jobs, max_jobs_multicut=max_jobs_mc,
             input_path=path, input_key=input_key,
             ws_path=path, ws_key=ws_key,
             problem_path=exp_path, node_labels_key=assignment_key,
             output_path=path, output_key=out_key,
             lifted_labels_path=path, lifted_labels_key=nucleus_seg_key,
             clear_labels_path=clear_path, clear_labels_key=clear_key,
             node_label_dict=node_label_dict,
             n_scales=1, lifted_prefix='nuclei', nh_graph_depth=3,
             sanity_checks=False, skip_ws=True)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Lifted Multicut failed")


def run_morphology(path, use_curated_affs, use_lmc, tmp_folder, target, max_jobs):

    task = MorphologyWorkflow
    stage = 'result'
    config_folder = os.path.join(tmp_folder, 'configs')
    prefix = 'lmc' if use_lmc else 'mc'

    if use_curated_affs:
        this_tmp = os.path.join(tmp_folder, 'tmp_%s_curated' % prefix)
        input_key = 'volumes/cells/curated_%s/%s' % (prefix, stage)
        task_prefix = 'curated_%s_%s' % (prefix, stage)
        prefix = 'curated_%s' % prefix
    else:
        this_tmp = os.path.join(tmp_folder, 'tmp_%s' % prefix)
        input_key = 'volumes/segmentation/%s/%s' % (prefix, stage)
        task_prefix = '%s_%s' % (prefix, stage)

    t = task(tmp_folder=this_tmp, max_jobs=max_jobs, target=target,
             config_dir=config_folder,
             input_path=path, input_key=input_key,
             output_path=path, output_key='morphology/%s/%s' % (prefix, stage),
             prefix=task_prefix, max_jobs_merge=32)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Morphology failed")


def map_nuclei(path, use_curated_affs,
               tmp_folder, target, max_jobs,
               max_overlap=False):
    task = NucleusAssignmentWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    prefix = 'lmc'
    identifier = 'result'
    nucleus_seg_key = 'volumes/nuclei/segmentation'

    if use_curated_affs:
        prefix = 'curated_%s' % prefix

    out_key = None
    this_tmp = os.path.join(tmp_folder, 'tmp_%s' % prefix)
    seg_key = 'volumes/cells/%s/%s' % (prefix, identifier)
    t = task(tmp_folder=this_tmp, max_jobs=max_jobs,
             target=target, config_dir=config_folder,
             path=path, seg_key=seg_key,
             nucleus_seg_key=nucleus_seg_key,
             output_key=out_key, prefix='%s_%s' % (prefix, identifier),
             max_overlap=max_overlap)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Map nuclei failed")


def unmerge_nuclei(path, use_curated_affs,
                   tmp_folder, target, max_jobs, max_threads):
    task = UnmergeWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    prefix = 'lmc'
    if use_curated_affs:
        prefix = 'curated_%s' % prefix

    tmp_unmerge = os.path.join(tmp_folder, 'tmp_%s' % prefix)
    exp_path = os.path.join(tmp_unmerge, 'problem.n5')
    assignment_key = 'node_labels/cells/%s/result' % prefix

    node_label_key = 'node_overlaps/nuclei'
    nucleus_mapping_key = 'nuclei_overlaps/%s_filtered_size' % prefix

    ws_key = 'volumes/cells/curated_watershed' if use_curated_affs else 'volumes/cells/watershed'
    out_key = 'volumes/cells/%s/filtered_unmerge' % prefix
    ass_out_key = 'node_labels/cells/%s/filtered_unmerge' % prefix

    configs = task.get_config()
    config = configs['fix_merges']
    config.update({'threads_per_job': max_threads, 'mem_limit': 256, 'time_limit': 240})
    with open('./configs/fix_merges.config', 'w') as f:
        json.dump(config, f)
    config = configs['find_merges']
    config.update({'mem_limit': 32})
    with open('./configs/find_merges.config', 'w') as f:
        json.dump(config, f)

    # clear the ids of the objects mapped to npil / cuticle
    clear_ids = [2, 3287]

    # we set the min nucleus overlap to ~ quarter of the median nucleus size
    min_nucleus_overlap = 25000

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             target=target, config_dir=config_folder,
             path=path, problem_path=exp_path, ws_key=ws_key,
             assignment_key=assignment_key, nucleus_mapping_key=nucleus_mapping_key,
             graph_key='s0/graph', features_key='s0/costs', node_label_key=node_label_key,
             ass_out_key=ass_out_key, out_key=out_key, clear_ids=clear_ids,
             min_overlap=min_nucleus_overlap)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Unmerge nuclei failed")


def filter_size(path, use_curated_affs, use_lmc, identifier,
                target, tmp_folder, max_jobs, max_threads):
    task = SizeFilterAndGraphWatershedWorkflow

    config_folder = os.path.join(tmp_folder, 'configs')
    prefix = 'lmc' if use_lmc else 'mc'
    if use_curated_affs:
        frag_key = 'volumes/cells/curated_watershed'
        prefix = 'curated_%s' % prefix
    else:
        frag_key = 'volumes/cells/watershed'

    seg_key = 'volumes/cells/%s/%s' % (prefix, identifier)
    assignment_key = 'node_labels/cells/%s/%s' % (prefix, identifier)

    out_key = 'volumes/cells/%s/filtered_size' % prefix
    ass_out_key = 'node_labels/cells/%s/filtered_size' % prefix

    this_tmp = os.path.join(tmp_folder, 'tmp_%s' % prefix)
    exp_path = os.path.join(this_tmp, 'problem.n5')

    config = task.get_config()['graph_watershed_assignments']
    config.update({'threads_per_job': max_threads, 'mem_limit': 256, 'time_limit': 180})
    with open(os.path.join(config_folder, 'graph_watershed_assignments.config'), 'w') as f:
        json.dump(config, f)

    # number of cells should be smaller than int16 max
    target_number = 32000
    t = task(tmp_folder=this_tmp, max_jobs=max_jobs,
             target=target, config_dir=config_folder,
             problem_path=exp_path, graph_key='s0/graph', features_key='s0/costs',
             path=path, segmentation_key=seg_key, fragments_key=frag_key,
             assignment_key=assignment_key,
             output_path=path, output_key=out_key,
             assignment_out_key=ass_out_key,
             relabel=True, from_costs=True, target_number=target_number)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Filter sizes failed")


def cell_segmentation_workflow(path, aff_path,
                               mask_path, mask_key,
                               region_path, region_key,
                               use_curated_affs, use_lmc,
                               tmp_folder, target, max_jobs,
                               roi_begin=None, roi_end=None):
    # number of jobs and threads for target
    assert target in ('slurm', 'local')
    if target == 'local':
        max_jobs_mc = 1
        max_threads = 16
    else:
        max_jobs_mc = 15
        max_threads = 8

    config_dir = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_dir, roi_begin, roi_end)

    run_watershed(path, aff_path, use_curated_affs,
                  mask_path, mask_key,
                  tmp_folder, target, max_jobs)
    if use_lmc:
        run_lmc(path, aff_path, use_curated_affs,
                region_path, region_key,
                tmp_folder, target,
                max_threads, max_jobs, max_jobs_mc)
    else:
        run_mc(path, aff_path, use_curated_affs,
               tmp_folder, target,
               max_threads, max_jobs, max_jobs_mc)

    # postprocessing:
    # 1.) compute sizes for size threshold
    run_morphology(path, use_curated_affs, use_lmc,
                   tmp_folder, target, max_jobs)

    identifier = 'result'
    # we unmerge only if we also use lmc, because this takes nuclei into account
    if use_lmc:
        # 2.) map nuclei to cells
        map_nuclei(path, use_curated_affs,
                   tmp_folder, target, max_jobs)
        # 3.) unmerge cells with more than one assigned nucleus
        unmerge_nuclei(path, use_curated_affs,
                       tmp_folder, target, max_jobs, max_threads)
        identifier = 'filtered_unmerge'

    # 4.) filter sizes with graph watershed
    filter_size(path, use_curated_affs, use_lmc, identifier,
                target, tmp_folder, max_jobs, max_threads)
