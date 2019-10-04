#! /bin/python

import os
import sys
import json

import luigi

import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask
from scripts.extension.attributes.vc_assignments_impl import vc_assignments as vc_assignments_impl

#
# Gene Attribute Tasks
#


class VCAssignmentsBase(luigi.Task):
    """ VCAssignments base class
    """

    task_name = 'vc_assignments'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    segmentation_path = luigi.Parameter()
    vc_volume_path = luigi.Parameter()
    vc_expression_path = luigi.Parameter()
    med_expression_path = luigi.Parameter()
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
                       'vc_volume_path': self.vc_volume_path,
                       'vc_expression_path': self.vc_expression_path,
                       'med_expression_path': self.med_expression_path,
                       'output_path': self.output_path})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class VCAssignmentsLocal(VCAssignmentsBase, LocalTask):
    """ VCAssignments on local machine
    """
    pass


class VCAssignmentsSlurm(VCAssignmentsBase, SlurmTask):
    """ VCAssignments on slurm cluster
    """
    pass


#
# Implementation
#

def vc_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    segmentation_path = config['segmentation_path']
    vc_volume_path = config['vc_volume_path']
    vc_expression_path = config['vc_expression_path']
    med_expression_path = config['med_expression_path']

    output_path = config['output_path']
    n_threads = config.get('threads_per_job', 1)

    vc_assignments_impl(segmentation_path, vc_volume_path, vc_expression_path,
                        med_expression_path, output_path, n_threads)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    vc_assignments(job_id, path)
