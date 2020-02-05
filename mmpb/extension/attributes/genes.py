#! /bin/python

import os
import sys
import json

import luigi
import numpy as np

import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask
from mmpb.extension.attributes.genes_impl import gene_assignments

#
# Gene Attribute Tasks
#


class GenesBase(luigi.Task):
    """ Genes base class
    """

    task_name = 'genes'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    segmentation_path = luigi.Parameter()
    segmentation_key = luigi.Parameter()
    genes_path = luigi.Parameter()
    labels_path = luigi.Parameter()
    output_path = luigi.Parameter()
    #
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'segmentation_path': self.segmentation_path,
                       'segmentation_key': self.segmentation_key,
                       'genes_path': self.genes_path,
                       'output_path': self.output_path,
                       'labels_path': self.labels_path})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class GenesLocal(GenesBase, LocalTask):
    """ Genes on local machine
    """
    pass


class GenesSlurm(GenesBase, SlurmTask):
    """ Genes on slurm cluster
    """
    pass


#
# Implementation
#

def genes(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    segmentation_path = config['segmentation_path']
    segmentation_key = config['segmentation_key']
    genes_path = config['genes_path']
    labels_path = config['labels_path']
    output_path = config['output_path']
    n_threads = config.get('threads_per_job', 1)

    labels = np.load(labels_path)
    gene_assignments(segmentation_path, segmentation_key,
                     genes_path, labels, output_path, n_threads)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    genes(job_id, path)
