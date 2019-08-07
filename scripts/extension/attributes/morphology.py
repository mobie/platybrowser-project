#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
import pandas as pd

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask
from scripts.extension.attributes.morphology_impl import morphology_impl

#
# Morphology Attribute Tasks
#


class MorphologyBase(luigi.Task):
    """ Morphology base class
    """

    task_name = 'morphology'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    segmentation_path = luigi.Parameter()
    in_table_path = luigi.Parameter()
    output_prefix = luigi.Parameter()
    # resolution of the segmentation at full scale
    resolution = luigi.ListParameter()
    # scales of segmentation and raw data used for the computation
    seg_scale = luigi.IntParameter()
    raw_scale = luigi.IntParameter(default=3)
    # prefix
    prefix = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    # minimum and maximum sizes for objects
    min_size = luigi.IntParameter()
    max_size = luigi.IntParameter(default=None)
    # path for cell nucleus mapping, that is used for additional
    # table filtering
    mapping_path = luigi.IntParameter(default='')
    # input path for intensity calcuation
    # if '', intensities will not be calculated
    raw_path = luigi.Parameter(default='')
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        # we hard-code the chunk-size to 1000 for now
        block_list = vu.blocks_in_volume([self.number_of_labels], [1000])

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'segmentation_path': self.segmentation_path,
                       'output_prefix': self.output_prefix,
                       'in_table_path': self.in_table_path,
                       'raw_path': self.raw_path,
                       'mapping_path': self.mapping_path,
                       'seg_scale': self.seg_scale,
                       'raw_scale': self.raw_scale,
                       'resolution': self.resolution,
                       'min_size': self.min_size,
                       'max_size': self.max_size})

        # prime and run the job
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, self.prefix)

    def output(self):
        out_path = os.path.join(self.tmp_folder,
                                '%s_%s.log' % (self.task_name, self.prefix))
        return luigi.LocalTarget(out_path)


class MorphologyLocal(MorphologyBase, LocalTask):
    """ Morphology on local machine
    """
    pass


class MorphologySlurm(MorphologyBase, SlurmTask):
    """ Morphology on slurm cluster
    """
    pass


#
# Implementation
#


def morphology(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    segmentation_path = config['segmentation_path']
    in_table_path = config['in_table_path']
    raw_path = config['raw_path']
    mapping_path = config['mapping_path']
    output_prefix = config['output_prefix']

    min_size = config['min_size']
    max_size = config['max_size']

    resolution = config['resolution']
    raw_scale = config['raw_scale']
    seg_scale = config['seg_scale']

    block_list = config['block_list']

    # read the base table
    table = pd.read_csv(in_table_path, sep='\t')

    # get the label ranges for this job
    n_labels = table.shape[0]
    blocking = nt.blocking([0], [n_labels], [1000])
    label_starts, label_stops = [], []
    for block_id in block_list:
        block = blocking.getBlock(block_id)
        label_starts.append(block.begin[0])
        label_stops.append(block.end[0])

    stats = morphology_impl(segmentation_path, raw_path, table, mapping_path,
                            min_size, max_size,
                            resolution, raw_scale, seg_scale,
                            label_starts, label_stops)

    output_path = output_prefix + '_job%i.csv' % job_id
    fu.log("Save result to %s" % output_path)
    stats.to_csv(output_path, index=False, sep='\t')
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    morphology(job_id, path)
