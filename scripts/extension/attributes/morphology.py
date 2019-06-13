#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import pandas as pd
import nifty.tools as nt
from skimage.measure import regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
from skimage.util import pad

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask

#
# Morphology Attribute Tasks
#


# FIXME something in here uses a lot of threads
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


# get shape of full data & downsampling factor
def get_scale_factor(path, key_full, key, resolution):
    with vu.file_reader(path, 'r') as f:
        full_shape = f[key_full].shape
        shape = f[key].shape

    # scale factor for downsampling
    scale_factor = [res * (fs / sh)
                    for res, fs, sh in zip(resolution,
                                           full_shape,
                                           shape)]
    return scale_factor


def filter_table(table, min_size, max_size):
    if max_size is None:
        table = table.loc[table['n_pixels'] >= min_size, :]
    else:
        criteria = np.logical_and(table['n_pixels'] > min_size, table['n_pixels'] < max_size)
        table = table.loc[criteria, :]
    return table


def filter_table_from_mapping(table, mapping_path):
    # read in numpy array of mapping of cells to nuclei - first column cell id, second nucleus id
    mapping = np.genfromtxt(mapping_path, skip_header=1, delimiter='\t')[:, :2].astype('uint64')
    # remove zero labels from this table too, if exist
    mapping = mapping[np.logical_and(mapping[:, 0] != 0,
                                     mapping[:, 1] != 0)]
    table = table.loc[np.isin(table['label_id'], mapping[:, 0]), :]
    return table


def load_data(ds, row, scale):
    # compute the bounding box from the row information
    mins = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
    maxs = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
    mins = [int(mi / sca) for mi, sca in zip(mins, scale)]
    maxs = [int(ma / sca) + 1 for ma, sca in zip(maxs, scale)]
    bb = tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))
    # load the data from the bounding box
    return ds[bb]


def morphology_row_features(mask, scale):

    # Calculate stats from skimage
    ski_morph = regionprops(mask.astype('uint8'))

    # volume in pixels
    volume_in_pix = ski_morph[0]['area']

    # extent
    extent = ski_morph[0]['extent']

    # The mesh calculation below fails if an edge of the segmentation is right up against the
    # edge of the volume - gives an open, rather than a closed surface
    # Pad by a few pixels to avoid this
    mask = pad(mask, 10, mode='constant')

    # surface area of mesh around object (other ways to calculate better?)
    verts, faces, normals, values = marching_cubes_lewiner(mask, spacing=tuple(scale))
    surface_area = mesh_surface_area(verts, faces)

    # volume in microns
    volume_in_microns = np.prod(scale)*volume_in_pix

    # sphericity (as in morpholibj)
    # Should run from zero to one
    sphericity = (36*np.pi*(float(volume_in_microns)**2))/(float(surface_area)**3)

    return [volume_in_microns, extent, surface_area, sphericity]


def intensity_row_features(raw, mask):
    intensity_vals_in_mask = raw[mask]
    # mean and stdev - use float64 to avoid silent overflow errors
    mean_intensity = np.mean(intensity_vals_in_mask, dtype=np.float64)
    st_dev = np.std(intensity_vals_in_mask, dtype=np.float64)
    return mean_intensity, st_dev


# compute morphology (and intensity features) for label range
def morphology_features_for_label_range(table, ds, ds_raw,
                                        scale_factor_seg, scale_factor_raw,
                                        label_begin, label_end):
    label_range = np.logical_and(table['label_id'] >= label_begin, table['label_id'] < label_end)
    sub_table = table.loc[label_range, :]
    stats = []
    for row in sub_table.itertuples(index=False):
        label_id = int(row.label_id)

        # load the segmentation data from the bounding box corresponding
        # to this row
        seg = load_data(ds, row, scale_factor_seg)

        # compute the segmentation mask and check that we have
        # foreground in the mask
        seg_mask = seg == label_id
        if seg_mask.sum() == 0:
            # if the seg mask is empty, we simply skip this label-id
            continue

        # compute the morphology features from the segmentation mask
        result = [float(label_id)] + morphology_row_features(seg_mask, scale_factor_seg)

        # compute the intensiry features from raw data and segmentation mask
        if ds_raw is not None:
            raw = load_data(ds_raw, row, scale_factor_raw)
            # resize the segmentation mask if it does not fit the raw data
            if seg_mask.shape != raw.shape:
                seg_mask = resize(seg_mask, raw.shape,
                                  order=0, mode='reflect',
                                  anti_aliasing=True, preserve_range=True).astype('bool')
            result += intensity_row_features(raw, seg_mask)
        stats.append(result)
    return stats


def compute_morphology_features(table, segmentation_path, raw_path,
                                seg_key, raw_key,
                                scale_factor_seg, scale_factor_raw,
                                blocking, block_list):

    if raw_path != '':
        assert raw_key is not None and scale_factor_raw is not None
        f_raw = vu.file_reader(raw_path, 'r')
        ds_raw = f_raw[raw_key]
    else:
        f_raw = ds_raw = None

    with vu.file_reader(segmentation_path, 'r') as f:
        ds = f[seg_key]

        stats = []
        for block_id in block_list:
            block = blocking.getBlock(block_id)
            label_a, label_b = block.begin[0], block.end[0]
            fu.log("Computing features from label-id %i to %i" % (label_a, label_b))
            stats.extend(morphology_features_for_label_range(table, ds, ds_raw,
                                                             scale_factor_seg, scale_factor_raw,
                                                             label_a, label_b))
    if f_raw is not None:
        f_raw.close()

    # convert to pandas table and add column names
    stats = pd.DataFrame(stats)
    columns = ['label_id',
               'shape_volume_in_microns', 'shape_extent', 'shape_surface_area', 'shape_sphericity']
    if raw_path != '':
        columns += ['intensity_mean', 'intensity_st_dev']
    stats.columns = columns
    return stats


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

    # keys to segmentation and raw data for the different scales
    seg_key_full = 't00000/s00/0/cells'
    seg_key = 't00000/s00/%i/cells' % seg_scale
    raw_key_full = 't00000/s00/0/cells'
    raw_key = 't00000/s00/%i/cells' % raw_scale

    # get scale factor for the segmentation
    scale_factor_seg = get_scale_factor(segmentation_path, seg_key_full, seg_key, resolution)

    # get scale factor for raw data (if it's given)
    if raw_path != '':
        fu.log("Have raw path; compute intensity features")
        # NOTE for now we can hard-code the resolution for the raw data here,
        # but we might need to change this if we get additional dataset(s)
        raw_resolution = [0.025, 0.01, 0.01]
        scale_factor_raw = get_scale_factor(raw_path, raw_key_full, raw_key, raw_resolution)
    else:
        fu.log("Don't have raw path; do not compute intensity features")
        raw_resolution = scale_factor_raw = None

    # read the base table
    table = pd.read_csv(in_table_path, sep='\t')
    n_labels = table.shape[0]
    fu.log("Initial number of labels: %i" % n_labels)
    # remove zero label if it exists
    table = table.loc[table['label_id'] != 0, :]

    # if we have a mappin, only keep objects in the mapping
    # (i.e cells that have assigned nuclei)
    if mapping_path != '':
        fu.log("Have mapping path %s" % mapping_path)
        table = filter_table_from_mapping(table, mapping_path)
        fu.log("Number of labels after filter with mapping: %i" % table.shape[0])
    # filter by size
    table = filter_table(table, min_size, max_size)
    fu.log("Number of labels after size filte: %i" % table.shape[0])

    blocking = nt.blocking([0], [n_labels], [1000])

    fu.log("Computing morphology features")
    stats = compute_morphology_features(table, segmentation_path, raw_path,
                                        seg_key, raw_key, scale_factor_seg, scale_factor_raw,
                                        blocking, block_list)

    output_path = output_prefix + '_job%i.csv' % job_id
    fu.log("Save result to %s" % output_path)
    stats.to_csv(output_path, index=False, sep='\t')
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    morphology(job_id, path)
