import numpy as np
import pandas as pd
import vigra
from elf.io import open_file, is_dataset
from .evaluate_annotations import evaluate_annotations, merge_evaluations


def get_bounding_box(ds):
    attrs = ds.attrs
    start, stop = attrs['starts'], attrs['stops']
    bb = tuple(slice(sta, sto) for sta, sto in zip(start, stop))
    return bb


def eval_slice(ds_seg, ds_ann, ignore_seg_ids, min_radius,
               return_masks=False):
    ds_seg.n_threads = 8
    ds_ann.n_threads = 8

    bb = get_bounding_box(ds_ann)
    annotations = ds_ann[:]
    seg = ds_seg[bb].squeeze().astype('uint32')
    assert annotations.shape == seg.shape

    seg_eval = vigra.analysis.labelImageWithBackground(seg)

    if ignore_seg_ids is None:
        this_ignore_ids = None
    else:
        ignore_mask = np.isin(seg, ignore_seg_ids)
        this_ignore_ids = np.unique(seg_eval[ignore_mask])

    fg_annotations = np.isin(annotations, [1, 2]).astype('uint32')
    bg_annotations = annotations == 3

    return evaluate_annotations(seg_eval, fg_annotations, bg_annotations,
                                this_ignore_ids, min_radius=min_radius,
                                return_masks=return_masks)


def get_ignore_seg_ids(table_path, ignore_names=['cuticle', 'neuropil', 'yolk']):
    table = pd.read_csv(table_path, sep='\t')
    ignore_seg_ids = []
    for name in ignore_names:
        col = table[name].values.astype('uint8')
        ignore_seg_ids.extend(np.where(col == 1)[0].tolist())
    ignore_seg_ids = np.unique(ignore_seg_ids)
    return ignore_seg_ids


def to_scores(eval_res):
    n = float(eval_res['n_annotations'] - eval_res['n_unmatched'])
    n_splits = eval_res['n_splits']
    n_merges = eval_res['n_merged_annotations']
    return n_merges / n, n_splits / n, n


def normal_eval(g, ds_seg, ignore_seg_ids, min_radius):
    eval_res = {}

    def visit_annotation(name, node):
        nonlocal eval_res
        if is_dataset(node):
            print("Evaluating:", name)
            res = eval_slice(ds_seg, node, ignore_seg_ids, min_radius)
            eval_res = merge_evaluations(res, eval_res)
            # for debugging
            # print("current eval:", eval_res)
        else:
            print("Group:", name)

    g.visititems(visit_annotation)
    return to_scores(eval_res)


# TODO
def semantic_eval(g, ds_seg, ignore_seg_ids, min_radius, semantic_mapping):
    pass


def eval_cells(seg_path, seg_key,
               annotation_path, annotation_key=None,
               ignore_seg_ids=None, min_radius=16,
               semantic_mapping=None):
    """ Evaluate the cell segmentation by computing
    the percentage of falsely merged and split cell annotations
    in manually annotated validation slices.
    """

    with open_file(seg_path, 'r') as f_seg, open_file(annotation_path, 'r') as f_ann:
        ds_seg = f_seg[seg_key]
        g = f_ann if annotation_key is None else f_ann[annotation_key]

        if semantic_mapping is None:
            return normal_eval(g, ds_seg, ignore_seg_ids, min_radius)
        else:
            return semantic_eval(g, ds_seg, ignore_seg_ids, min_radius,
                                 semantic_mapping)
