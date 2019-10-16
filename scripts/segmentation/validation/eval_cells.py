import numpy as np
import pandas as pd
import vigra
from elf.io import open_file, is_dataset
from .evaluate_annotations import evaluate_annotations, merge_evaluations


def eval_slice(ds_seg, ds_ann, ignore_seg_ids, min_radius):
    ds_seg.n_threads = 8
    ds_ann.n_threads = 8

    attrs = ds_ann.attrs
    start, stop = attrs['starts'], attrs['stops']
    bb = tuple(slice(sta, sto) for sta, sto in zip(start, stop))

    annotations = ds_ann[:]
    seg = ds_seg[bb].squeeze()

    seg_eval = vigra.analysis.labelImageWithBackground(seg)

    if ignore_seg_ids is None:
        this_ignore_ids = None
    else:
        ignore_mask = np.isin(seg, ignore_seg_ids)
        this_ignore_ids = np.unique(seg_eval[ignore_mask])

    fg_annotations = np.isin(annotations, [1, 2]).astype('uint32')
    bg_annotations = annotations == 3

    return evaluate_annotations(seg_eval, fg_annotations, bg_annotations,
                                this_ignore_ids, min_radius=min_radius)


def get_ignore_seg_ids(table_path, ignore_names=['cuticle', 'neuropil', 'yolk']):
    table = pd.read_csv(table_path)
    ignore_seg_ids = []
    for name in ignore_names:
        col = table[name].values.astype('uint8')
        ignore_seg_ids.extend(np.where(col == 1)[0].tolist())
    ignore_seg_ids = np.unique(ignore_seg_ids)
    return ignore_seg_ids


def eval_cells(seg_path, seg_key,
               annotation_path, annotation_key,
               ignore_seg_ids=None, min_radius=16):
    """ Evaluate the cell segmentation.
    """

    eval_res = {}
    with open_file(seg_path, 'r') as f_seg, open_file(annotation_path) as f_ann:
        ds_seg = f_seg[seg_key]
        g = f_ann[annotation_key]

        def visit_annotation(name, node):
            nonlocal eval_res
            if is_dataset(node):
                res = eval_slice(ds_seg, node, ignore_seg_ids, min_radius)
                eval_res = merge_evaluations(res, eval_res)

        g.visititems(visit_annotation)

    return eval_res
