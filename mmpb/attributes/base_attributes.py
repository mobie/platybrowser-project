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
            os.remove(output_path)
    elif os.path.exists(output_path) and not override:
        return

    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    # we have two different versions of the id mapping:
    # the old one that only saves the mapped ids
    # and the new one that also saves the mapped counts
    # in the second case, we use the counts to decide the labeling for
    # mapping ids that result from several merged previous ids
    id_mapping = {int(k): v for k, v in id_mapping.items()}
    if isinstance(id_mapping[0], list):
        mapping_counts = np.array([v[1] for v in id_mapping.values()])
        id_mapping = {k: v[0] for k, v in id_mapping.items()}
    else:
        mapping_counts = None

    assert os.path.exists(table_path), table_path
    table = pd.read_csv(table_path, sep='\t')
    id_col = table[column_name].values
    id_col[np.isnan(id_col)] = 0
    id_col = id_col.astype('uint32')

    # use mapping counts to decide the mapped ids for merges
    if mapping_counts is not None:

        mapping_keys = np.array([int(key) for key in id_mapping.keys()])
        mapping_values = np.array([int(val) for val in id_mapping.values()])
        unique_vals, val_counts = np.unique(mapping_values, return_counts=True)
        merged_ids = unique_vals[val_counts > 1]
        if merged_ids[0] == 0:
            merged_ids = merged_ids[1:]

        # this could be sped up with np.unique tricks, but I don't expect there to be many merged ids
        # between versions, so this should not matter for now
        keep_mask = np.ones(len(id_col), dtype='bool')
        for merged_id in merged_ids:
            id_mask = mapping_values == merged_id
            source_ids = mapping_keys[id_mask]
            ids_sorted = np.argsort(mapping_counts[id_mask])[::-1]
            drop_ids = source_ids[ids_sorted[1:]]
            keep_mask[np.isin(id_col, drop_ids)] = False

        columns = table.columns
        table = table.values
        table = table[keep_mask]
        table = pd.DataFrame(table, columns=columns)
        id_col = id_col[keep_mask]
        assert len(table) == len(id_col)

    # map values for the id col
    id_col = nt.takeDict(id_mapping, id_col)
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
