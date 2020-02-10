# implementation for the tasks in 'genes.py', can be called standalone
import csv
from concurrent import futures

import numpy as np
from elf.io import open_file
from vigra.analysis import extractRegionFeatures
from vigra.sampling import resize


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


def gene_assignments(segmentation_path, segmentation_key,
                     genes_path, labels, output_path,
                     n_threads):
    """ Write a table with genes assigned to segmentation by overlap.

    Arguments:
        segmentation_path [str] - path to hdf5 file with the cell segmentation
        segmentation_key [str] - path in file to the segmentation dataset.
        genes_path [str] - path to hdf5 file with spatial gene expression.
            We expect the datasets 'genes' and 'gene_names' to be present.
        labels [np.ndarray] - cell id labels
        output_path [str] - where to write the result table
        n_threads [int] - number of threads used for the computation
    """

    with open_file(segmentation_path, 'r') as f:
        segmentation = f[segmentation_key][:]

    genes_dset = 'genes'
    names_dset = 'gene_names'
    with open_file(genes_path, 'r') as f:
        ds = f[genes_dset]
        gene_shape = ds.shape[1:]
        all_genes = ds[:]
        gene_names = [i.decode('utf-8') for i in f[names_dset]]

    # resize the segmentation to gene space
    segmentation = resize(segmentation.astype("float32"),
                          shape=gene_shape, order=0).astype('uint16')
    print("Compute gene expression ...")
    avail_labels, expression = get_cell_expression(segmentation, all_genes, n_threads)

    print('Save results to %s' % output_path)
    write_genes_table(output_path, expression, gene_names, labels, avail_labels)


# TODO write argument parser
if __name__ == '__main__':
    pass
