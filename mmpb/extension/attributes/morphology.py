#! /bin/python

import os
import sys
import json
from math import ceil

import luigi
import nifty.tools as nt
import pandas as pd

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask
from mmpb.extension.attributes.morphology_impl import (morphology_impl_cell,
                                                       morphology_impl_nucleus)

#
# Morphology Attribute Tasks
#


class MorphologyBase(luigi.Task):
    """ Morphology base class
    """

    task_name = 'morphology'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # compute cell features or nucleus features?
    compute_cell_features = luigi.BoolParameter()

    # paths to raw data and segmentations
    # if the raw path is None, we don't compute intensity features
    raw_path = luigi.Parameter(default=None)
    # we always need the nucleus segmentation
    nucleus_segmentation_path = luigi.Paramter()
    # we only need the cell segmentation if we compute cell morphology features
    cell_segmentation_path = luigi.Parameter(default=None)
    # we only need the chromatin segmentation if we compute nucleus features
    chromatin_segmentation_path = luigi.Paramter(default=None)

    # the scale used for computation, relative to the raw scale
    scale = luigi.IntParameter(default=3)

    # the input tables paths for the default table, the
    # nucleus mapping table and the region mapping table
    in_table_path = luigi.Parameter()
    # only need the mapping paths for the nucleus features
    nucleus_mapping_path = luigi.Paramter(default=None)
    region_mapping_path = luigi.Paramter(default=None)

    # prefix for the output tables
    output_prefix = luigi.Parameter()

    # minimum and maximum sizes for objects / bounding box
    min_size = luigi.IntParameter()
    max_size = luigi.IntParameter(default=None)
    max_bb = luigi.IntParameter()

    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def _update_config_for_cells(self, config):
        # check the relevant inputs for the cell morphology
        assert self.cell_segmentation_path is not None
        assert self.nucleus_mapping_path is not None
        assert self.region_mapping_path is not None
        config.update({'cell_segmentation_path': self.cell_segmentation_path,
                       'nucleus_segmentation_path': self.nucleus_segmentation_path,
                       'raw_path': self.raw_path,
                       'output_prefix': self.output_prefix,
                       'in_table_path': self.in_table_path,
                       'nucleus_mapping_path': self.nucleus_mapping_path,
                       'region_mapping_path': self.region_mapping_path,
                       'scale': self.scale, 'max_bb': self.max_bb,
                       'min_size': self.min_size, 'max_size': self.max_size})
        return config

    def _update_config_for_nuclei(self, config):
        # check the relevant inputs for the nucleus morphology
        assert self.chromatin_segmentation_path is not None
        assert self.raw_path is not None
        config.update({'nucleus_segmentation_path': self.nucleus_segmentation_path,
                       'chromatin_segmentation_path': self.chromatin_segmentation_path,
                       'raw_path': self.raw_path,
                       'output_prefix': self.output_prefix,
                       'in_table_path': self.in_table_path,
                       'scale': self.scale, 'max_bb': self.max_bb,
                       'min_size': self.min_size, 'max_size': self.max_size})
        return config

    def _get_number_of_labels(self):
        seg_path = self.cell_segmentation_path if self.compute_cell_features else\
            self.nucleus_segmentation_path
        key = 't00000/s00/0/cells'
        with vu.file_reader(seg_path, 'r') as f:
            n_labels = int(f[key].attrs['maxId']) + 1
        return n_labels

    def _compute_block_len(self, number_of_labels):
        ids_per_job = int(ceil(float(number_of_labels) / self.max_jobs))
        return ids_per_job

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        if self.compute_cell_features:
            config = self._update_config_for_cells(config)
        else:
            config = self._update_config_for_nuclei(config)

        # TODO match block size and number of blocks
        # we hard-code the chunk-size to 1000 for now
        number_of_labels = self._get_number_of_labels()
        block_len = self._compute_block_len(number_of_labels)
        block_list = vu.blocks_in_volume([number_of_labels], [block_len])
        config.update({'block_len': block_len,
                       'compute_cell_features': self.compute_cell_features,
                       'number_of_labels': number_of_labels})

        prefix = 'cells' if self.compute_cell_features else 'nuclei'
        # prime and run the job
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, prefix)

    def output(self):
        prefix = 'cells' if self.compute_cell_features else 'nuclei'
        out_path = os.path.join(self.tmp_folder,
                                '%s_%s.log' % (self.task_name, prefix))
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


def _morphology_nuclei(config, table, label_start, label_stop):
    # paths to raw data, nucleus segmentation and chromatin segmentation
    raw_path = config['raw_path']
    nucleus_segmentation_path = config['nucleus_segmentation_path']
    chromatin_segmentation_path = config['chromatin_segmentation_path']
    # minimal / maximal size and maximal bounding box volume
    min_size, max_size, max_bb = config['min_size'], config['max_size'], config['max_bb']
    # scale for nucleus segmentation, raw data and chromatin segmentation
    raw_scale = config['scale']
    nucleus_scale = raw_scale - 3
    chromatin_scale = raw_scale - 1  # TODO is this correct?
    # nucleus and chromatin resolution are hard-coded for now
    nucleus_resolution = [0.1, 0.08, 0.08]
    chromatin_resolution = [0.025, 0.02, 0.02]  # TODO is this correct?
    stats = morphology_impl_nucleus(nucleus_segmentation_path, raw_path,
                                    chromatin_segmentation_path,
                                    table, min_size, max_size, max_bb,
                                    nucleus_resolution, chromatin_resolution,
                                    nucleus_scale, raw_scale, chromatin_scale,
                                    label_start, label_stop)
    return stats


def _morphology_cells(config, table, label_start, label_stop):
    # paths to raw data, nucleus segmentation and chromatin segmentation
    raw_path = config['raw_path']
    cell_segmentation_path = config['cell_segmentation_path']
    nucleus_segmentation_path = config['nucleus_segmentation_path']
    # minimal / maximal size and maximal bounding box volume
    min_size, max_size, max_bb = config['min_size'], config['max_size'], config['max_bb']
    # scale for nucleus segmentation, raw data and chromatin segmentation
    raw_scale = config['scale']
    nucleus_scale = raw_scale - 3
    cell_scale = raw_scale - 1  # TODO is this correct?
    # nucleus and chromatin resolution are hard-coded for now
    nucleus_resolution = [0.1, 0.08, 0.08]
    cell_resolution = [0.025, 0.02, 0.02]  # TODO is this correct?
    # mapping from cells to nuclei and from cells to regions
    nucleus_mapping_path = config['mapping_path']
    region_mapping_path = config['region_mapping_path']
    stats = morphology_impl_cell(cell_segmentation_path, raw_path,
                                 nucleus_segmentation_path,
                                 table, nucleus_mapping_path,
                                 region_mapping_path,
                                 min_size, max_size, max_bb,
                                 cell_resolution, nucleus_resolution,
                                 cell_scale, raw_scale, nucleus_scale,
                                 label_start, label_stop)
    return stats


def morphology(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    # read the base table
    in_table_path = config['in_table_path']
    table = pd.read_csv(in_table_path, sep='\t')

    # determine the start and stop label for this job
    block_list = config['block_list']
    block_len = config['block_len']
    assert len(block_list) == 1, "Expected a single block, got %i" % len(block_list)

    # get the label range for this job
    n_labels = table.shape[0]
    blocking = nt.blocking([0], [n_labels], [block_len])
    block = blocking.getBlock(block_list[0])
    label_start, label_stop = block.begin, block.end

    # do we compute cell or nucleus features ?
    compute_cell_features = config['compute_cell_features']

    if compute_cell_features:
        fu.log("Compute morphology features for cells")
        stats = _morphology_cells(config, table, label_start, label_stop)
    else:
        fu.log("Compute morphology features for nuclei")
        stats = _morphology_nuclei(config, table, label_start, label_stop)

    # write the result
    output_prefix = config['output_prefix']
    output_path = output_prefix + '_job%i.csv' % job_id
    fu.log("Save result to %s" % output_path)
    stats.to_csv(output_path, index=False, sep='\t')
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    morphology(job_id, path)
