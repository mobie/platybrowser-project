import os
import json
import luigi

from cluster_tools import MulticutSegmentationWorkflow, LiftedMulticutSegmentationWorkflow
from cluster_tools.watershed import WatershedWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow

from mmpb.default_config import write_default_global_config
# special workflows
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
        tmp_ws = os.path.join(tmp_folder, 'curated_ws')
    else:
        aff_key = 'volumes/cells/affinities/s1'
        output_key = 'volumes/cells/watershed'
        tmp_ws = os.path.join(tmp_folder, 'curated_ws')

    task = WatershedWorkflow(tmp_folder=tmp_ws, config_dir=config_folder,
                             max_jobs=max_jobs, target=target,
                             input_path=aff_path, input_key=aff_key,
                             mask_path=path, mask_key=mask_key,
                             output_path=path, output_key=output_key,
                             two_pass=False)
    ret = luigi.build([task], local_scheduler=True)
    if not ret:
        raise RuntimeError("Watershed failed")


def run_mc(use_curated_affs, max_jobs, max_jobs_mc, max_threads, target):
    config_folder = './configs'
    path = PATH
    if use_curated_affs:
        input_key = 'volumes/curated_affinities/s1'
        ws_key = 'volumes/segmentation/curated_watershed'
        assignment_key = 'node_labels/curated_mc/result'
        out_key = 'volumes/segmentation/curated_mc/result'
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_mc_curated')
        exp_path = os.path.join(EXP_ROOT, 'mc_curated.n5')
    else:
        input_key = 'volumes/affinities/s1'
        ws_key = 'volumes/segmentation/watershed'
        assignment_key = 'node_labels/mc/result'
        out_key = 'volumes/segmentation/mc/result'
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_mc')
        exp_path = os.path.join(EXP_ROOT, 'mc.n5')

    configs = MulticutSegmentationWorkflow.get_config()
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

    task = MulticutSegmentationWorkflow(tmp_folder=tmp_folder, config_dir=config_folder, target=target,
                                        max_jobs=max_jobs, max_jobs_multicut=max_jobs_mc,
                                        input_path=path, input_key=input_key,
                                        ws_path=path, ws_key=ws_key,
                                        problem_path=exp_path,
                                        node_labels_key=assignment_key,
                                        output_path=path, output_key=out_key,
                                        n_scales=1,
                                        sanity_checks=False,
                                        skip_ws=True)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Multicut failed"


def run_lmc(use_curated_affs, max_jobs, max_jobs_mc, max_threads, target):
    config_folder = './configs'
    path = PATH
    nucleus_seg_key = 'volumes/nuclei/mws_mc_biased_filtered'
    if use_curated_affs:
        input_key = 'volumes/curated_affinities/s1'
        ws_key = 'volumes/segmentation/curated_watershed'
        assignment_key = 'node_labels/curated_lmc/result'
        out_key = 'volumes/segmentation/curated_lmc/result'
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_lmc_curated')
        exp_path = os.path.join(EXP_ROOT, 'lmc_curated.n5')

        clear_path = path
        clear_key = 'volumes/tissue/for_insert/s3'
        node_label_dict = {'ignore_transition': (path, clear_key)}
    else:
        input_key = 'volumes/affinities/s1'
        ws_key = 'volumes/segmentation/watershed'
        assignment_key = 'node_labels/lmc/result'
        out_key = 'volumes/segmentation/lmc/result'
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_lmc')
        exp_path = os.path.join(EXP_ROOT, 'lmc.n5')

        clear_path = clear_key = node_label_dict = None

    configs = LiftedMulticutSegmentationWorkflow.get_config()
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

    task = LiftedMulticutSegmentationWorkflow(tmp_folder=tmp_folder, config_dir=config_folder, target=target,
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
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Lifted Multicut failed"


def run_morphology(use_curated_affs, use_lmc, max_jobs, target, stage):
    path = PATH
    config_folder = './configs'
    prefix = 'lmc' if use_lmc else 'mc'
    if use_curated_affs:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_curated' % prefix)
        input_key = 'volumes/segmentation/curated_%s/%s' % (prefix, stage)
        task_prefix = 'curated_%s_%s' % (prefix, stage)
        prefix = 'curated_%s' % prefix
    else:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s' % prefix)
        input_key = 'volumes/segmentation/%s/%s' % (prefix, stage)
        task_prefix = '%s_%s' % (prefix, stage)
    task = MorphologyWorkflow(tmp_folder=tmp_folder, max_jobs=max_jobs, target=target,
                              config_dir=config_folder,
                              input_path=path, input_key=input_key,
                              output_path=path, output_key='morphology/%s/%s' % (prefix, stage),
                              prefix=task_prefix, max_jobs_merge=32)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Morphology failed"


def filter_size(size_threshold, use_curated_affs, use_lmc,
                max_jobs, max_threads, target):
    path = PATH
    config_folder = './configs'
    prefix = 'lmc' if use_lmc else 'mc'
    if use_curated_affs:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_curated_size_filter' % prefix)
        exp_path = os.path.join(EXP_ROOT, '%s_curated.n5' % prefix)

        seg_key = 'volumes/segmentation/curated_%s/result' % prefix
        frag_key = 'volumes/segmentation/curated_watershed'
        assignment_key = 'node_labels/curated_%s/result' % prefix

        out_key = 'volumes/segmentation/curated_%s/filtered_size' % prefix
        ass_out_key = 'node_labels/curated_%s/filtered_size' % prefix
    else:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_size_filter' % prefix)
        exp_path = os.path.join(EXP_ROOT, '%s.n5' % prefix)

        seg_key = 'volumes/segmentation/%s/result' % prefix
        frag_key = 'volumes/segmentation/watershed'
        assignment_key = 'node_labels/%s/result' % prefix

        out_key = 'volumes/segmentation/%s/filtered_size' % prefix
        ass_out_key = 'node_labels/%s/filtered_size' % prefix

    task = SizeFilterAndGraphWatershedWorkflow
    config = task.get_config()['graph_watershed_assignments']
    config.update({'threads_per_job': max_threads, 'mem_limit': 256, 'time_limit': 180, 'qoc': 'high'})
    with open(os.path.join(config_folder, 'graph_watershed_assignments.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             target=target, config_dir=config_folder,
             problem_path=exp_path, graph_key='s0/graph', features_key='s0/costs',
             path=path, segmentation_key=seg_key, fragments_key=frag_key,
             assignment_key=assignment_key,
             output_path=path, output_key=out_key,
             assignment_out_key=ass_out_key,
             size_threshold=size_threshold, relabel=True,
             from_costs=True)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Filter sizes failed"


def map_nuclei(use_curated_affs, use_lmc, max_jobs, max_threads, target, identifier,
               max_overlap=False):
    path = PATH
    config_folder = './configs'
    prefix = 'lmc' if use_lmc else 'mc'
    nucleus_seg_key = 'volumes/nuclei/mws_mc_biased_filtered'

    if use_curated_affs:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_curated_map_%s' % (prefix, identifier))
        seg_key = 'volumes/segmentation/curated_%s/%s' % (prefix, identifier)
        # out_key = 'volumes/segmentation/curated_%s_filtered' % prefix
        out_key = None
        prefix = 'curated_%s' % prefix
    else:
        tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_map' % prefix)
        seg_key = 'volumes/segmentation/%s/%s' % (prefix, identifier)
        # out_key = 'volumes/segmentation/%s_filtered' % prefix
        out_key = None

    task = NucleusAssignmentWorkflow(tmp_folder=tmp_folder, max_jobs=max_jobs,
                                     target=target, config_dir=config_folder,
                                     path=path, seg_key=seg_key,
                                     nucleus_seg_key=nucleus_seg_key,
                                     output_key=out_key, prefix='%s_%s' % (prefix, identifier),
                                     max_overlap=max_overlap)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Map nuclei failed"


def unmerge_nuclei(use_curated_affs, use_lmc, max_jobs, max_threads, target, min_size):
    path = PATH
    config_folder = './configs'
    prefix = 'lmc' if use_lmc else 'mc'
    if use_curated_affs:
        prefix = 'curated_%s' % prefix

    tmp_folder = os.path.join(TMP_ROOT, 'tmp_%s_unmerge' % prefix)
    exp_path = os.path.join(EXP_ROOT, '%s.n5' % prefix)
    assignment_key = 'node_labels/%s/filtered_size' % prefix

    node_label_key = 'node_overlaps/nuclei'
    nucleus_mapping_key = 'nuclei_overlaps/%s_filtered_size' % prefix

    ws_key = 'volumes/segmentation/curated_watershed' if use_curated_affs\
        else 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/%s/filtered_unmerge' % prefix
    ass_out_key = 'node_labels/%s/filtered_unmerge' % prefix

    configs = UnmergeWorkflow.get_config()
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

    task = UnmergeWorkflow(tmp_folder=tmp_folder, max_jobs=max_jobs,
                           target=target, config_dir=config_folder,
                           path=path, problem_path=exp_path, ws_key=ws_key,
                           assignment_key=assignment_key, nucleus_mapping_key=nucleus_mapping_key,
                           graph_key='s0/graph', features_key='s0/costs', node_label_key=node_label_key,
                           ass_out_key=ass_out_key, out_key=out_key, clear_ids=clear_ids,
                           min_overlap=min_nucleus_overlap, min_size=min_size)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Unmerge nuclei failed"


# TODO need to accept path + bounding box
# TODO move insert affinities to it's own function
# def cell_segmentation_workflow(use_curated_affs, use_lmc, target, max_jobs):
def cell_segmentation_workflow(path, aff_path,
                               mask_path, mask_key,
                               use_lmc, use_curated_affs,
                               tmp_folder, target, max_jobs):
    # number of jobs and threads for target
    assert target in ('slurm', 'local')
    if target == 'local':
        max_jobs_mc = 1
        max_threads = 16
    else:
        max_jobs_mc = 15
        max_threads = 8

    config_dir = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_dir)

    run_watershed(path, aff_path, use_curated_affs,
                  mask_path, mask_key,
                  tmp_folder, target, max_jobs)
    if use_lmc:
        run_lmc(use_curated_affs, max_jobs, max_jobs_mc, max_threads, target)
    else:
        run_mc(use_curated_affs, max_jobs, max_jobs_mc, max_threads, target)

    # postprocessing:
    # 1.) compute sizes for size threshold
    run_morphology(use_curated_affs, run_lmc, max_jobs, target, 'result')

    # we unmerge only if we also use lmc, because this takes nuclei into account
    if use_lmc:
        # 3.) map nuclei to cells
        map_nuclei(use_curated_affs, use_lmc, max_jobs, max_threads, target, 'filtered_size')
        # 4.) unmerge cells with more than one assigned nucleus
        unmerge_nuclei(use_curated_affs, use_lmc, max_jobs, max_threads, target, size_threshold)

    # TODO use num segment size threshold
    # size threshold to fit into uint16
    # size_threshold = 27020
    # size threshold to fit into int16
    size_threshold = 88604

    # 5.) filter sizes with graph watershed
    filter_size(size_threshold, use_curated_affs, use_lmc, max_jobs, max_threads, target)
