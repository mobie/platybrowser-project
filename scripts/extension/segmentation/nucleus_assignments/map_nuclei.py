#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import z5py
import nifty.distributed as ndist

import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class MapNucleiBase(luigi.Task):
    """ MapNuclei base class
    """

    task_name = 'map_nuclei'
    src_file = os.path.abspath(__file__)

    path = luigi.Parameter()
    key = luigi.Parameter()
    output_key = luigi.Parameter()
    res_path = luigi.Parameter()
    n_labels = luigi.IntParameter()
    assignment_threshold = luigi.Parameter(default=None)
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # we don't need any additional config besides the paths
        config = {"path": self.path, "key": self.key, "output_key": self.output_key,
                  "res_path": self.res_path, "assignment_threshold": self.assignment_threshold,
                  "n_labels": self.n_labels}

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MapNucleiLocal(MapNucleiBase, LocalTask):
    """
    MapNuclei on local machine
    """
    pass


class MapNucleiSlurm(MapNucleiBase, SlurmTask):
    """
    MapNuclei on slurm cluster
    """
    pass


class MapNucleiLSF(MapNucleiBase, LSFTask):
    """
    MapNuclei on lsf cluster
    """
    pass


#
# Implementation
#


def map_nuclei(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    path = config['path']
    key = config['key']
    assignment_threshold = config['assignment_threshold']
    n_labels = config['n_labels']

    output_key = config['output_key']
    res_path = config['res_path']

    ds = z5py.File(path)[key]
    n_chunks = ds.number_of_chunks

    overlaps = {}
    for chunk_id in range(n_chunks):
        cdict, _ = ndist.deserializeOverlapChunk(os.path.join(path, key),
                                                 (chunk_id,))
        overlaps.update(cdict)

    cell_ids = []
    nucleus_ids = []

    keys = overlaps.keys()
    for k in sorted(keys):
        if k == 0:
            continue
        ovlp = overlaps[k]
        nids = list(ovlp.keys())
        counts = list(ovlp.values())
        counts_tot = float(sum(counts))
        counts = [cnt / counts_tot for cnt in counts]

        max_id = np.argmax(counts)
        max_count = counts[max_id]

        if assignment_threshold is None:
            cell_ids.append(k)
            nucleus_ids.append(nids[max_id])
        else:
            if max_count > assignment_threshold:
                cell_ids.append(k)
                nucleus_ids.append(nids[max_id])
    assert len(cell_ids) == len(nucleus_ids)
    mapping = np.concatenate([np.array(cell_ids)[:, None], np.array(nucleus_ids)[:, None]], axis=1).astype('uint64')
    assert mapping.ndim == 2, str(mapping.shape)
    assert mapping.shape[1] == 2, str(mapping.shape)

    # serialize the mapping from cell ids to nucleus ids
    # and the cell ids that will be discarded
    with z5py.File(path) as f:
        chunks = (min(100000, len(mapping)), 1)
        ds_out = f.require_dataset(output_key, shape=mapping.shape, compression='gzip',
                                   chunks=chunks, dtype='uint64')
        ds_out[:] = mapping

    discard_ids = list(set(range(n_labels)) - set(cell_ids))
    with open(res_path, 'w') as f:
        json.dump(discard_ids, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    map_nuclei(job_id, path)
