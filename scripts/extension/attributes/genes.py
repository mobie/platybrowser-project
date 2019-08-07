#! /bin/python

import os
import sys
import json
import csv
from concurrent import futures

import luigi
import numpy as np
from vigra.analysis import extractRegionFeatures
from vigra.sampling import resize

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask

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
    gene_shape = luigi.ListParameter()
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
                       'labels_path': self.labels_path,
                       'gene_shape': self.gene_shape})

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


def get_sizes_and_bbs(data):
    # compute the relevant vigra region features
    features = extractRegionFeatures(data.astype('float32'), data.astype('uint32'),
                                     features=['Coord<Maximum >', 'Coord<Minimum >', 'Count'])

    # extract sizes from features
    cell_sizes = features['Count'].squeeze().astype('uint64')

    # compute bounding boxes from features
    mins = features['Coord<Minimum >'].astype('uint32')
    maxs = features['Coord<Maximum >'].astype('uint32') + 1
    cell_bbs = [tuple(slice(mi, ma) for mi, ma in zip(min_, max_))
                for min_, max_ in zip(mins, maxs)]
    return cell_sizes, cell_bbs


def get_cell_expression(segmentation, all_genes, n_threads):
    num_genes = all_genes.shape[0]
    # NOTE we need to recalculate the unique labels here, beacause we might not
    # have all labels due to donwsampling
    labels = np.unique(segmentation)
    cells_expression = np.zeros((len(labels), num_genes), dtype='float32')
    cell_sizes, cell_bbs = get_sizes_and_bbs(segmentation)

    def compute_expressions(cell_idx, cell_label):
        # get size and boundinng box of this cell
        cell_size = cell_sizes[cell_label]
        bb = cell_bbs[cell_label]
        # get the cell mask and the gene expression in bounding box
        cell_masked = segmentation[bb] == cell_label
        genes_in_cell = all_genes[(slice(None),) + bb]
        # accumulate the gene expression channels over the cell mask
        gene_expr_sum = np.sum(genes_in_cell[:, cell_masked] > 0, axis=1)
        # divide by the cell size and write result
        cells_expression[cell_idx] = gene_expr_sum / cell_size

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(compute_expressions, cell_idx, cell_label)
                 for cell_idx, cell_label in enumerate(labels) if cell_label != 0]
        [t.result() for t in tasks]
    return labels, cells_expression


def write_genes_table(output_path, expression, gene_names, labels, avail_labels):
    n_labels = len(labels)
    n_cols = len(gene_names) + 1

    data = np.zeros((n_labels, n_cols), dtype='float32')
    data[:, 0] = labels
    data[avail_labels, 1:] = expression

    col_names = ['label_id'] + gene_names
    assert data.shape[1] == len(col_names)
    with open(output_path, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(col_names)
        csv_writer.writerows(data)


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
    gene_shape = tuple(config['gene_shape'])
    n_threads = config.get('threads_per_job', 1)

    fu.log("Loading segmentation, labels and gene-data")
    # load segmentation, labels and genes
    with vu.file_reader(segmentation_path, 'r') as f:
        segmentation = f[segmentation_key][:]
    labels = np.load(labels_path)

    genes_dset = 'genes'
    names_dset = 'gene_names'
    with vu.file_reader(genes_path, 'r') as f:
        all_genes = f[genes_dset][:]
        gene_names = [i.decode('utf-8') for i in f[names_dset]]

    # resize the segmentation to gene space
    segmentation = resize(segmentation.astype("float32"),
                          shape=gene_shape, order=0).astype('uint16')
    fu.log("Compute gene expression")
    avail_labels, expression = get_cell_expression(segmentation, all_genes, n_threads)

    fu.log('Save results to %s' % output_path)
    write_genes_table(output_path, expression, gene_names, labels, avail_labels)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    genes(job_id, path)
