import os
import json
import numpy as np
import pandas as pd
from elf.io import open_file
from pybdv.util import get_key

import luigi
import nifty.tools as nt
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.morphology import RegionCentersWorkflow
from .util import write_csv
from ..default_config import write_default_global_config
from ..util import is_h5_file


def n5_attributes(input_path, input_key, tmp_folder, target, max_jobs):
    task = MorphologyWorkflow

    out_path = os.path.join(tmp_folder, 'data.n5')
    config_folder = os.path.join(tmp_folder, 'configs')

    out_key = 'attributes'
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs, target=target,
             config_dir=config_folder,
             input_path=input_path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix='attributes', max_jobs_merge=min(32, max_jobs))
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Attribute workflow failed")
    return out_path, out_key


# set the anchor to region center (= maximum of boundary distance transform
# inside the object) instead of com
def run_correction(input_path, input_key,
                   tmp_folder, target, max_jobs):
    task = RegionCentersWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    out_path = os.path.join(tmp_folder, 'data.n5')
    out_key = 'region_centers'

    # FIXME anchor correction still takes very long at this scale,
    # maybe should switch to s5
    # we need to run this at a lower scale, as a heuristic,
    # we take the first scale with all dimensions < 1750 pix
    # (corresponds to scale 4 in sbem)
    max_dim_size = 1750
    scale_key = input_key
    is_h5 = is_h5_file(input_path)
    with open_file(input_path, 'r') as f:
        while True:
            shape = f[scale_key].shape
            if all(sh <= max_dim_size for sh in shape):
                break

            try:
                scale = int(scale_key.split('/')[2]) + 1
            except ValueError:
                scale = int(scale_key.split('/')[-1][1:]) + 1
            next_scale_key = get_key(is_h5, time_point=0, setup_id=0, scale=scale)
            if next_scale_key not in f:
                break
            scale_key = next_scale_key

    with open_file(input_path, 'r') as f:
        shape1 = f[input_key].shape
        shape2 = f[scale_key].shape
    scale_factor = np.array([float(sh1) / sh2 for sh1, sh2 in zip(shape1, shape2)])

    config = task.get_config()['region_centers']
    config.update({'time_limit': 180, 'mem_limit': 32})
    with open(os.path.join(config_folder, 'region_centers.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=input_path, input_key=scale_key,
             output_path=out_path, output_key=out_key,
             ignore_label=0)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Anchor correction failed")

    with open_file(out_path, 'r') as f:
        anchors = f[out_key][:]
    anchors *= scale_factor
    return anchors


def to_csv(input_path, input_key, output_path, resolution,
           anchors=None):
    # load the attributes from n5
    with open_file(input_path, 'r') as f:
        attributes = f[input_key][:]
    label_ids = attributes[:, 0:1]

    # the colomn names
    col_names = ['label_id',
                 'anchor_x', 'anchor_y', 'anchor_z',
                 'bb_min_x', 'bb_min_y', 'bb_min_z',
                 'bb_max_x', 'bb_max_y', 'bb_max_z',
                 'n_pixels']

    # we need to switch from our axis conventions (zyx)
    # to java conventions (xyz)
    res_in_micron = resolution[::-1]
    # reshuffle the attributes to fit the output colomns

    def translate_coordinate_tuple(coords):
        coords = coords[:, ::-1]
        for d in range(3):
            coords[:, d] *= res_in_micron[d]
        return coords

    # center of mass / anchor points
    com = attributes[:, 2:5]
    if anchors is None:
        anchors = translate_coordinate_tuple(com)
    else:
        assert len(anchors) == len(com)
        assert anchors.shape[1] == 3

        # some of the corrected anchors might not be present,
        # so we merge them with the com here
        invalid_anchors = np.isclose(anchors, 0.).all(axis=1)
        anchors[invalid_anchors] = com[invalid_anchors]
        anchors = translate_coordinate_tuple(anchors)

    # attributes[5:8] = min coordinate of bounding box
    minc = translate_coordinate_tuple(attributes[:, 5:8])
    # attributes[8:11] = min coordinate of bounding box
    maxc = translate_coordinate_tuple(attributes[:, 8:11])

    # NOTE: attributes[1] = size in pixel
    # make the output attributes
    data = np.concatenate([label_ids, anchors, minc, maxc, attributes[:, 1:2]], axis=1)
    write_csv(output_path, data, col_names)


def base_attributes(input_path, input_key, output_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=True):

    # prepare cluster tools tasks
    write_default_global_config(os.path.join(tmp_folder, 'configs'))

    # make base attributes as n5 dataset
    tmp_path, tmp_key = n5_attributes(input_path, input_key,
                                      tmp_folder, target, max_jobs)

    # correct anchor positions
    if correct_anchors:
        anchors = run_correction(input_path, input_key,
                                 tmp_folder, target, max_jobs)
    else:
        anchors = None

    # write output to csv
    to_csv(tmp_path, tmp_key, output_path, resolution, anchors)

    # load and return label_ids
    with open_file(tmp_path, 'r') as f:
        label_ids = f[tmp_key][:, 0]
    return label_ids


def write_additional_table_file(table_folder):
    # get all the file names in the table folder
    file_names = os.listdir(table_folder)
    file_names.sort()

    # make sure we have the default table
    default_name = 'default.csv'
    if default_name not in file_names:
        raise RuntimeError("Did not find the default table ('default.csv') in the table folder %s" % table_folder)

    # don't write anything if we don't have additional tables
    if len(file_names) == 1:
        return

    # write file for the additional tables
    out_file = os.path.join(table_folder, 'additional_tables.txt')
    with open(out_file, 'w') as f:
        for name in file_names:
            ext = os.path.splitext(name)[1]
            # only add csv files
            if ext != '.csv':
                continue
            # don't add the default table
            if name == 'default.csv':
                continue
            f.write(name + '\n')


# TODO implement merge rules
# TODO use propagate_ids from mmpb.util
def propagate_attributes(id_mapping_path, table_path, output_path,
                         column_name, merge_rule=None, override=False):
    """ Propagate id column to new ids.
    """
    # if the output already exists, we assume that the propagation
    # was already done and we just continue
    if os.path.exists(output_path) and override:
        if os.path.islink(output_path):
            os.unlink(output_path)
        else:
            raise RuntimeError("Cannot override file.")
    elif os.path.exists(output_path) and not override:
        return

    print(id_mapping_path)
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)
    id_mapping = {int(k): v for k, v in id_mapping.items()}

    assert os.path.exists(table_path), table_path
    table = pd.read_csv(table_path, sep='\t')
    id_col = table[column_name].values
    id_col[np.isnan(id_col)] = 0
    id_col = id_col.astype('uint32')

    # keys = list(id_mapping.keys())
    id_col = nt.takeDict(id_mapping, id_col)

    # TODO need to implement merge rules
    # to update values for multiple ids that were mapped to the same value
    # (this is necessary if we update a label_id column, for which the values should be unique)
    # new_ids, new_id_counts = np.unique(id_col, return_counts=True)
    # merge_ids = new_ids[]

    table[column_name] = id_col
    table.to_csv(output_path, index=False, sep='\t')


# Do we need to extend the cell criterion? Possibilties:
# - size threshold
def add_cell_criterion_column(base_table_path, nucleus_mapping_path, out_table_path=None):
    """ Add a column to the cell defaults table that indicates whether the id
        is considered as a cell or not.

        Currently the criterion is based on having a unique nucleus id mapped to the cell.
    """
    base_table = pd.read_csv(base_table_path, sep='\t')
    # skip if we already have the cells column
    if 'cells' in base_table.columns:
        return

    nucleus_mapping = pd.read_csv(nucleus_mapping_path, sep='\t')
    assert len(base_table) == len(nucleus_mapping)

    mapped_nucleus_ids = nucleus_mapping['nucleus_id'].values
    mapped, mapped_counts = np.unique(mapped_nucleus_ids, return_counts=True)

    unique_mapped_nuclei = mapped[mapped_counts == 1]
    cell_criterion = np.isin(mapped_nucleus_ids, unique_mapped_nuclei)
    assert len(cell_criterion) == len(base_table)

    base_table['cells'] = cell_criterion.astype('uint8')

    out_path = base_table_path if out_table_path is None else out_table_path
    base_table.to_csv(out_path, index=False, sep='\t')
