from math import ceil, floor
import vigra
from elf.io import open_file, is_dataset
from .evaluate_annotations import evaluate_annotations, merge_evaluations


def get_bounding_box(ds, scale_factor):
    attrs = ds.attrs
    start, stop = attrs['starts'], attrs['stops']
    bb = tuple(slice(int(floor(sta / scale_factor)),
                     int(ceil(sto / scale_factor))) for sta, sto in zip(start, stop))
    return bb


def to_scores(eval_res):
    n = float(eval_res['n_annotations'] - eval_res['n_unmatched'])
    fp = eval_res['n_splits']
    fn = eval_res['n_merged_annotations']
    return fp / n, fn / n, n


# we may want to do a max projection of some z context ?!
def get_nucleus_segmentation(ds_seg, bb):
    seg = ds_seg[bb].squeeze().astype('uint32')
    return seg


# need to downsample the annotations and bounding box to fit the
# nucleus segmentation
def eval_slice(ds_seg, ds_ann, min_radius, return_masks=False):
    ds_seg.n_threads = 8
    ds_ann.n_threads = 8

    bb = get_bounding_box(ds_ann, scale_factor=4.)
    annotations = ds_ann[:]
    seg = get_nucleus_segmentation(ds_seg, bb)
    annotations = vigra.sampling.resize(annotations.astype('float32'),
                                        shape=seg.shape, order=0).astype('uint32')

    fg_annotations = (annotations == 1).astype('uint32')
    bg_annotations = None

    return evaluate_annotations(seg, fg_annotations, bg_annotations,
                                min_radius=min_radius, return_masks=return_masks)


def eval_nuclei(seg_path, seg_key,
                annotation_path, annotation_key=None,
                min_radius=6):
    """ Evaluate the nucleus segmentation by computing
    the percentage of false positive and false negative nucleus annotations
    in manually annotated validation slices.
    """
    eval_res = {}
    with open_file(seg_path, 'r') as f_seg, open_file(annotation_path, 'r') as f_ann:
        ds_seg = f_seg[seg_key]
        g = f_ann if annotation_key is None else f_ann[annotation_key]

        def visit_annotation(name, node):
            nonlocal eval_res
            if is_dataset(node):
                print("Evaluating:", name)
                res = eval_slice(ds_seg, node, min_radius)
                eval_res = merge_evaluations(res, eval_res)
                # for debugging
                # print("current eval:", eval_res)
            else:
                print("Group:", name)

        g.visititems(visit_annotation)

    return to_scores(eval_res)
