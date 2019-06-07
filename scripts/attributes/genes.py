import csv
import h5py
import numpy as np
from vigra.analysis import extractRegionFeatures
from vigra.sampling import resize


# TODO
# wrap this in a cluster_tools task in order to run remotely
# fix blatant inefficiencis (size loop)
# make test to check against original table


def get_bbs(data):
    num_cells = (np.max(data)).astype('int') + 1
    cells_bbs = [[] for i in range(num_cells)]
    mins_and_maxs = extractRegionFeatures(data.astype('float32'), data.astype('uint32'),
                                          features=['Coord<Maximum >', 'Coord<Minimum >'])
    mins = mins_and_maxs['Coord<Minimum >'].astype('uint32')
    maxs = mins_and_maxs['Coord<Maximum >'].astype('uint32') + 1
    for cell in range(num_cells):
        cell_bb = []
        cell_min = mins[cell]
        cell_max = maxs[cell]
        for axis in range(3):
            cell_bb.append(slice(cell_min[axis], cell_max[axis]))
        cells_bbs[cell] = tuple(cell_bb)
    return cells_bbs


# TODO very inefficient, can use "Count" feature of vigra features instead
# and then just do this in `get_bbs` as well
def get_cell_sizes(data):
    max_label = (np.max(data)).astype('uint32')
    cell_sizes = [0] * (max_label + 1)
    Z, X, Y = data.shape
    for z in range(Z):
        for x in range(X):
            for y in range(Y):
                label = data[z, x, y]
                cell_sizes[label] += 1
    cell_sizes = np.array(cell_sizes)
    return cell_sizes


def get_cell_expression(segm_data, all_genes):
    num_genes = all_genes.shape[0]
    labels = list(np.unique(segm_data))
    cells_expression = np.zeros((len(labels), num_genes), dtype='float32')
    cell_sizes = get_cell_sizes(segm_data)
    cell_bbs = get_bbs(segm_data)
    for cell_idx in range(len(labels)):
        cell_label = labels[cell_idx]
        if cell_label == 0:
            continue
        cell_size = cell_sizes[cell_label]
        bb = cell_bbs[cell_label]
        cell_masked = (segm_data[bb] == cell_label)
        genes_in_cell = all_genes[tuple([slice(0, None)] + list(bb))]
        for gene in range(num_genes):
            gene_expr = genes_in_cell[gene]
            gene_expr_sum = np.sum(gene_expr[cell_masked] > 0)
            cells_expression[cell_idx, gene] = gene_expr_sum / cell_size
    return labels, cells_expression


def write_genes_table(segm_file, genes_file, table_file, labels):
    dset = 't00000/s00/4/cells'
    new_shape = (570, 518, 550)
    genes_dset = 'genes'
    names_dset = 'gene_names'

    with h5py.File(segm_file, 'r') as f:
        segment_data = f[dset][:]

    # TODO loading the whole thing into ram takes a lot of memory
    with h5py.File(genes_file, 'r') as f:
        all_genes = f[genes_dset][:]
        gene_names = [i.decode('utf-8') for i in f[names_dset]]

    num_genes = len(gene_names)
    downsampled_data = resize(segment_data.astype("float32"), shape=new_shape, order=0).astype('uint16')
    avail_labels, expression = get_cell_expression(downsampled_data, all_genes)

    with open(table_file, 'w') as genes_table:
        csv_writer = csv.writer(genes_table, delimiter='\t')
        _ = csv_writer.writerow(['label_id'] + gene_names)
        for label in labels:
            if label in avail_labels:
                idx = avail_labels.index(label)
                _ = csv_writer.writerow([label] + list(expression[idx]))
            else:
                _ = csv_writer.writerow([label] + [0] * num_genes)
