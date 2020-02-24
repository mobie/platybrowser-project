#! /bin/python

import os
import json
import sys

import luigi
import z5py
import nifty.distributed as ndist

import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class FindMergesBase(luigi.Task):
    task_name = 'find_merges'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path = luigi.Parameter()
    key = luigi.Parameter()
    out_path = luigi.Parameter()
    clear_ids = luigi.ListParameter()
    min_overlap = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        config.update({'path': self.path, 'key': self.key,
                       'clear_ids': self.clear_ids,
                       'out_path': self.out_path, 'min_overlap': self.min_overlap})

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class FindMergesLocal(FindMergesBase, LocalTask):
    """
    FindMerges on local machine
    """
    pass


class FindMergesSlurm(FindMergesBase, SlurmTask):
    """
    FindMerges on slurm cluster
    """
    pass


class FindMergesLSF(FindMergesBase, LSFTask):
    """
    FindMerges on lsf cluster
    """
    pass


def find_merges(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    path = config['path']
    key = config['key']
    clear_ids = config['clear_ids']
    min_overlap = config['min_overlap']
    out_path = config['out_path']

    # load the overlap dictionary
    n_chunks = z5py.File(path)[key].number_of_chunks
    overlaps = {}
    for chunk_id in range(n_chunks):
        cdict, _ = ndist.deserializeOverlapChunk(path, key, (chunk_id,))
        overlaps.update(cdict)

    # find objects with merges according to the min overlap
    merge_objects = {}
    for object_id, nucleus_overlaps in overlaps.items():
        if object_id in clear_ids:
            continue
        # get nucleus_ids and normalized overlaps
        nucleus_ids = list(nucleus_overlaps.keys())
        this_overlaps = list(nucleus_overlaps.values())

        mapped_nuclei = [nid for nid, ovlp in zip(nucleus_ids, this_overlaps)
                         if (ovlp > min_overlap and nid != 0)]
        if len(mapped_nuclei) > 1:
            merge_objects[object_id] = mapped_nuclei

    fu.log("Found %i objects with merges" % len(merge_objects))
    with open(out_path, 'w') as f:
        json.dump(merge_objects, f)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    find_merges(job_id, path)
