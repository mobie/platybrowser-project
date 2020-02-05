#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import vigra
import luigi
import nifty
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Multicut Tasks
#


class FixMergesBase(luigi.Task):
    """ FixMerges base class
    """

    task_name = 'fix_merges'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path = luigi.Parameter()
    problem_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()
    node_label_key = luigi.Parameter()
    merge_object_path = luigi.Parameter()
    out_key = luigi.Parameter()
    from_costs = luigi.BoolParameter()
    relabel = luigi.BoolParameter()

    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        config = self.get_task_config()
        config.update({'path': self.path, 'problem_path': self.problem_path,
                       'assignment_key': self.assignment_key, 'graph_key': self.graph_key,
                       'features_key': self.features_key, 'node_label_key': self.node_label_key,
                       'out_key': self.out_key, 'from_costs': self.from_costs, 'relabel': self.relabel,
                       'merge_object_path': self.merge_object_path})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class FixMergesLocal(FixMergesBase, LocalTask):
    """ FixMerges on local machine
    """
    pass


class FixMergesSlurm(FixMergesBase, SlurmTask):
    """ FixMerges on slurm cluster
    """
    pass


class FixMergesLSF(FixMergesBase, LSFTask):
    """ FixMerges on lsf cluster
    """
    pass


#
# Implementation
#


def fix_merge_assignments(graph, assignments, merge_objects,
                          node_labels, features, n_threads):

    # json turns all keys to string
    merge_objects = {int(k): v for k, v in merge_objects.items()}

    uv_ids = graph.uvIds()

    def fix_object(object_id):
        # find the nodes corresponding to this object
        node_ids = np.where(assignments == object_id)[0].astype('uint64')

        # extract the subgraph corresponding to this object
        # we allow for invalid nodes here,
        # which can occur for un-connected graphs resulting from bad masks ...
        inner_edges, _ = graph.extractSubgraphFromNodes(node_ids, allowInvalidNodes=True)
        sub_uvs = uv_ids[inner_edges]

        # relanbel the sub-nodes / edges
        nodes_relabeled, max_id, mapping = vigra.analysis.relabelConsecutive(node_ids,
                                                                             start_label=0,
                                                                             keep_zeros=False)
        sub_uvs = nt.takeDict(mapping, sub_uvs)
        n_local_nodes = max_id + 1

        # make sub-graph and get the edge costs
        sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
        sub_graph.insertEdges(sub_uvs)
        sub_features = features[inner_edges]
        assert len(sub_features) == sub_graph.numberOfEdges

        # get the seeds from the mapped nuclei
        sub_node_labels = node_labels[node_ids]
        nucleus_ids = merge_objects[object_id]

        seeds = np.zeros(n_local_nodes, dtype='uint64')
        for seed_id, n_id in enumerate(nucleus_ids):
            has_seed = sub_node_labels == n_id
            seeds[has_seed] = seed_id + 1

        # check that we have at least two seeds present
        # note that we can have discrepencies here due to differences in mapping strategies
        if len(np.unique(seeds)) < 3:
            return None

        # resolve by graph watershed
        sub_result = nifty.graph.edgeWeightedWatershedsSegmentation(sub_graph, seeds, sub_features)
        return sub_result

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(fix_object, object_id)
                 for object_id in merge_objects]
        results = [t.result() for t in tasks]

    # Filter for objects that could not be resolved
    object_ids = [object_id for object_id, res
                  in zip(merge_objects.keys(), results) if res is not None]
    fu.log("Can resolve %i of %i initial objects" % (len(object_ids), len(results)))
    results = [res for res in results if res is not None]
    results = {object_id: res for object_id, res in zip(object_ids, results)}
    merge_objects = {object_id: merge_objects[object_id] for object_id in object_ids}

    # make the id offset
    full_offset = int(assignments.max()) + 1
    offsets = np.array([len(v) for v in merge_objects.values()], dtype='uint64')
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    offsets += full_offset
    offsets = {object_id: offset for object_id, offset
               in zip(merge_objects.keys(), offsets)}

    # insert the sub-solutions
    for object_id in merge_objects:
        node_ids = np.where(assignments == object_id)
        sub_result = results[object_id] + offsets[object_id]
        assignments[node_ids] = sub_result

    return assignments


def fix_merges(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    path = config['path']
    problem_path = config['problem_path']
    merge_object_path = config['merge_object_path']

    assignment_key = config['assignment_key']
    out_key = config['out_key']

    graph_key = config['graph_key']
    features_key = config['features_key']
    node_label_key = config['node_label_key']

    from_costs = config['from_costs']
    relabel = config['relabel']
    n_threads = config['threads_per_job']

    # load the merge objects
    with open(merge_object_path) as f:
        merge_objects = json.load(f)

    if len(merge_objects) == 0:
        fu.log("no merges to resolve")
        ln_src = os.path.join(path, assignment_key)
        ln_dst = os.path.join(path, out_key)
        os.symlink(ln_src, ln_dst)
        return

    fu.log("resolving %i merges" % len(merge_objects))

    fu.log("reading problem from %s" % problem_path)
    f = vu.file_reader(path)
    problem = vu.file_reader(problem_path, 'r')

    # load the graph
    fu.log("reading graph from path in problem: %s" % graph_key)
    graph = ndist.Graph(os.path.join(problem_path, graph_key),
                        numberOfThreads=n_threads)

    # load the assignments
    ds = f[assignment_key]
    chunks = ds.chunks
    ds.n_threads = 8
    assignments = ds[:]

    # load the costs
    ds = problem[features_key]
    ds.n_threads = 8
    if ds.ndim == 2:
        features = ds[:, 0].squeeze()
    else:
        features = ds[:]
    if from_costs:
        minc = features.min()
        fu.log("Mapping costs with range %f to %f to range 0 to 1" % (minc, features.max()))
        features -= minc
        features /= features.max()
        features = 1. - features

    # load the node labels
    ds = problem[node_label_key]
    ds.n_threads = n_threads
    node_labels = ds[:]

    assignments = fix_merge_assignments(graph, assignments, merge_objects,
                                        node_labels, features, n_threads)

    # relabel and save assignments
    if relabel:
        vigra.analysis.relabelConsecutive(assignments, out=assignments, start_label=1, keep_zeros=True)
    ds = f.create_dataset(out_key, shape=assignments.shape, chunks=chunks,
                          dtype='uint64', compression='gzip')
    ds.n_threads = 8
    ds[:] = assignments

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    fix_merges(job_id, path)
