# implementation for the tasks in 'vc_assignments.py', can be called standalone
import os
import csv
from concurrent import futures

import argparse
import numpy as np
from elf.io import open_file
from pybdv.util import get_key
from pybdv.metadata import get_data_path
from vigra.analysis import extractRegionFeatures
from vigra.sampling import resize
from vigra.filters import distanceTransform


def get_n5(path):
    xml_file = os.path.splitext(path)[0] + '.xml'
    return get_data_path(xml_file, return_absolute_path=True)


def get_common_genes(profile_file, ov_expression_file):
    with open(ov_expression_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        med_gene_names = csv_reader.fieldnames[1:]
    cells_gene_expression = np.loadtxt(ov_expression_file, delimiter='\t',
                                       skiprows=1)[:, 1:]
    med_gene_indices = []
    vc_gene_indices = []
    common_gene_names = []

    # get the names of genes used for vc's
    with open(profile_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        vc_gene_names = csv_reader.fieldnames

    # find a subset of genes both used for vc's and available as MEDs
    for i, name in enumerate(vc_gene_names):
        if name in med_gene_names:
            med_gene_indices.append(med_gene_names.index(name))
            vc_gene_indices.append(i)
            common_gene_names.append(name)

    # from expression_by_overlap assignment extract only the subset genes
    cells_expression_subset = np.take(cells_gene_expression, med_gene_indices,
                                      axis=1)
    # from vcs_expression extract only the subset genes
    vc_expression_subset = np.loadtxt(profile_file, delimiter='\t',
                                      skiprows=1, usecols=vc_gene_indices)
    # add the null vc with no expression
    vc_expression_subset = np.insert(vc_expression_subset, 0,
                                     np.zeros(len(vc_gene_indices)),
                                     axis=0)
    print(len(common_gene_names), 'common genes found in VCs and MEDs')
    return cells_expression_subset, vc_expression_subset, common_gene_names


def get_bbs(data, offset):
    shape = np.array(data.shape)
    # compute the relevant vigra region features
    # beware: for the absent labels the results are ridiculous
    features = extractRegionFeatures(data.astype('float32'), data.astype('uint32'),
                                     features=['Coord<Maximum >', 'Coord<Minimum >'])
    # compute bounding boxes from features
    mins = features['Coord<Minimum >'] - offset
    maxs = features['Coord<Maximum >'] + offset + 1
    # to prevent 'out of range' due to offsets
    mins[np.where(mins < 0)] = 0
    exceed_bounds = np.where(maxs > shape)
    maxs[exceed_bounds] = shape[exceed_bounds[1]]
    # get a bb for each cell
    cell_bbs = [tuple(slice(mi, ma) for mi, ma in zip(min_, max_))
                for min_, max_ in zip(np.uint32(mins), np.uint32(maxs))]
    return cell_bbs


def get_distances(em_data, vc_data, cells_expression, vc_expression, n_threads,
                  offset=10):
    num_cells = cells_expression.shape[0]
    # some labels might be lost due to downsampling
    avail_cells = np.unique(em_data)
    num_vcs = np.max(vc_data) + 1
    distance_matrix = np.full((num_cells, num_vcs), np.nan)
    bbs = get_bbs(em_data, offset)

    def get_distance(cell):
        if cell == 0:
            return

        bb = bbs[cell]
        cell_mask = (em_data[bb] == cell).astype("uint32")
        dist = distanceTransform(cell_mask)
        vc_roi = vc_data[bb]
        vc_candidate_list = np.unique(vc_roi).astype('int')
        vc_list = [vc for vc in vc_candidate_list
                   if np.min(dist[vc_roi == vc]) <= offset]
        cell_genes = cells_expression[cell]
        if 0 not in vc_list:
            vc_list = np.append(vc_list, 0)
        vc_genes = vc_expression[vc_list]
        # calculate the genetic distance between the cell and the vcs
        distance = np.sum(np.abs(cell_genes - vc_genes), axis=1)
        distance_matrix[cell][vc_list] = distance

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(get_distance, cell_id) for cell_id in avail_cells]
        [t.result() for t in tasks]

    return distance_matrix


def assign_vc(distances, vc_expression):
    num_cells = distances.shape[0]
    # assign to 0 if no vcs were found at all
    assignments = [0 if np.all(np.isnan(distances[cell]))
                   else np.nanargmin(distances[cell])
                   for cell in range(num_cells)]
    cells_expr = vc_expression[assignments]
    expr_sum = np.sum(cells_expr, axis=1)
    assign_and_sum = np.column_stack((assignments, expr_sum))
    cells_expr = np.insert(cells_expr, 0, np.arange(num_cells), axis=1)
    expr_assign_sum = np.hstack((cells_expr, assign_and_sum))
    return expr_assign_sum


def vc_assignments(segm_volume_file, em_dset,
                   vc_volume_file, cm_dset, vc_expr_file,
                   cells_med_expr_table, output_gene_table, n_threads):
    # volume file for vc's (generated from CellModels_coordinates)
    with open_file(vc_volume_file, 'r') as f:
        vc_data = f[cm_dset][:]

    # downsample segmentation data to the same resolution as gene data
    with open_file(segm_volume_file, 'r') as f:
        segm_data = f[em_dset][:]
    downsampled_segm_data = resize(segm_data.astype("float32"), shape=vc_data.shape,
                                   order=0).astype('uint16')

    # get the genes that were both used for vcs and are in med files
    cells_expression_subset, vc_expression_subset, common_gene_names = \
        get_common_genes(vc_expr_file, cells_med_expr_table)

    # get the genetic distance from cells to surrounding vcs
    dist_matrix = get_distances(downsampled_segm_data, vc_data,
                                cells_expression_subset,
                                vc_expression_subset, n_threads)

    # assign the cells to the genetically closest vcs
    cell_assign = assign_vc(dist_matrix, vc_expression_subset)
    # write down a new table
    col_names = ['label_id'] + common_gene_names + ['VC', 'expression_sum']
    assert cell_assign.shape[1] == len(col_names)
    with open(output_gene_table, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(col_names)
        csv_writer.writerows(cell_assign)


if __name__ == '__main__':

    platy_data_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
    ov_table_path = 'tables/sbem-6dpf-1-whole-segmented-cells/genes.csv'
    vc_volume_path = 'images/local/prospr-6dpf-1-whole-virtual-cells.n5'
    vc_profile_path = 'tables/prospr-6dpf-1-whole-virtual-cells/profile_clust_curated.csv'
    segm_path = 'images/local/sbem-6dpf-1-whole-segmented-cells.n5'

    parser = argparse.ArgumentParser(description='Assign cells to genetically closest VCs')

    parser.add_argument('version', type=str,
                        help='the version of platy data')
    parser.add_argument('output_file', type=str,
                        help='the files with cell expression assigned by VC')
    parser.add_argument('--vc_volume_file', type=str, default=None,
                        help='the h5 file with VC labels')
    parser.add_argument('--vc_profile_file', type=str, default=None,
                        help='table of expression by VC')
    args = parser.parse_args()

    gene_table_file = os.path.join(platy_data_path, args.version, ov_table_path)
    segment_file = get_n5(os.path.join(platy_data_path, args.version, segm_path))
    vc_volume_file = get_n5(os.path.join(platy_data_path, args.version, vc_volume_path)) \
                     if args.vc_volume_file == None else args.vc_volume_file
    vc_profile_file = os.path.join(platy_data_path, args.version, vc_profile_path) \
                      if args.vc_profile_file == None else args.vc_profile_file
    output_file = args.output_file

    # number of threads hard-coded for now
    n_threads = 8
    em_dset = get_key(segment_file.endswith('.h5'), 0, 0, 4)
    cm_dset = get_key(vc_volume_file.endswith('.h5'), 0, 0, 0)
    vc_assignments(segment_file, em_dset,
                   vc_volume_file, cm_dset, vc_profile_file,
                   gene_table_file, output_file, n_threads)
