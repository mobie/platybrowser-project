import csv
import h5py
import numpy as np
from vigra.analysis import extractRegionFeatures
from vigra.sampling import resize


# TODO
# wrap this in a cluster_tools task in order to run remotely
# fix blatant inefficiencis (size loop)
# make test to check against original table


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


def get_cell_expression(segm_data, all_genes):
    num_genes = all_genes.shape[0]
    labels = list(np.unique(segm_data))
    cells_expression = np.zeros((len(labels), num_genes), dtype='float32')
    cell_sizes, cell_bbs = get_sizes_and_bbs(segm_data)
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
