import numpy as np
import vigra
from tqdm import tqdm


def merge_evaluations(trgt, src):
    for name, val in trgt.items():
        if name in src:
            src[name] += val
        else:
            src[name] = val
    return src


def get_radii(seg):
    # TODO I am not sure if this is the best measure.
    # maybe use estimate based on convex hull instead?
    # compute and filter by region radii
    radii = vigra.analysis.extractRegionFeatures(seg.astype('float32'), seg,
                                                 features=['RegionRadii'])['RegionRadii']
    radii = radii.min(axis=1)
    return radii


def evaluate_annotations(seg, fg_annotations, bg_annotations,
                         ignore_seg_ids=None, min_radius=16,
                         return_masks=False, return_ids=False):
    """ Evaluate segmentation based on evaluations.
    """

    # apply connected components to the foreground annotations
    # NOTE we don't apply ccs to the segmentation, because this there is
    # cases where this is not quite appropriate
    labels = vigra.analysis.labelImageWithBackground(fg_annotations)

    # get the seg ids and label ids
    seg_ids = np.unique(seg)[1:]
    label_ids = np.unique(labels)[1:]
    radii = get_radii(seg)

    # categories for the segmentation objects:
    # unmatched: segments with no corresponding annotation
    # matched: segments matched to one annotation
    # overmatched: sements matched to multiple annotation
    unmatched_ids = []
    matched_ids = {}
    overmatched_ids = {}

    # iterate over all seg-ids and map them to annotations
    n_segments = 0
    for seg_id in tqdm(seg_ids):
        mask = seg == seg_id

        # check if this is an ignore id and skip
        if ignore_seg_ids is not None and seg_id in ignore_seg_ids:
            continue
        has_bg_label = bg_annotations[mask].sum() > 0

        # find the overlapping label ids
        this_labels = np.unique(labels[mask])
        if 0 in this_labels:
            this_labels = this_labels[1:]

        # no labels -> this seg-id is unmatched and part of a false split,
        # unless we have overlap with a background annotation
        # or are in the filter ids
        if this_labels.size == 0:
            if not has_bg_label and radii[seg_id] > min_radius:
                unmatched_ids.append(seg_id)

        # one label -> this seg-id seems to be well matched
        # note that it could still be part of a false split, which we check later
        elif this_labels.size == 1:
            matched_ids[seg_id] = this_labels[0]

        # multiple labels -> this seg-id is over-matched and part of a false merge
        else:
            overmatched_ids[seg_id] = this_labels.tolist()

        # increase the segment count
        n_segments += 1

    # false splits = unmatched seg-ids and seg-ids corresponding to annotations
    # that were matched more than once

    # first, turn matched ids and labels into numpy arrays
    matched_labels = list(matched_ids.values())
    matched_ids = np.array(list(matched_ids.keys()), dtype='uint32')

    # find the unique matched labels, their counts and the mapping to the original array
    matched_labels, inv_mapping, matched_counts = np.unique(matched_labels, return_inverse=True, return_counts=True)
    matched_counts = matched_counts[inv_mapping]
    assert len(matched_counts) == len(matched_ids)

    # combine unmatched ids and ids matched more than once
    unmatched_ids = np.array(unmatched_ids, dtype='uint32')
    false_split_ids = np.concatenate([unmatched_ids, matched_ids[matched_counts > 1]])

    # false merge annotations = overmatched ids
    false_merge_ids = list(overmatched_ids.keys())
    false_merge_labels = np.array([lab for overmatched in overmatched_ids.values()
                                   for lab in overmatched], dtype='uint32')

    # find label ids that were not matched
    all_matched = np.concatenate([matched_labels, false_merge_labels])
    all_matched = np.unique(all_matched)
    unmatched_labels = np.setdiff1d(label_ids, all_matched)

    # print("Number of false splits:", len(false_split_ids), '/', n_segments)
    # print("Number of false merges:", len(false_merge_ids), '/', n_segments)
    # print("Number of unmatched labels:", len(unmatched_labels), '/', len(label_ids))
    metrics = {'n_annotations': len(label_ids), 'n_segments': n_segments,
               'n_splits': len(false_split_ids),
               'n_merged_annotations': len(false_merge_labels),
               'n_merged_ids': len(false_merge_ids),
               'n_unmatched': len(unmatched_labels)}

    if not return_masks and not return_ids:
        return metrics

    ret = (metrics,)
    if return_masks:
        fs_mask = np.isin(seg, false_split_ids).astype('uint32')
        fm_mask = np.isin(seg, false_merge_ids).astype('uint32')
        masks = {'splits': fs_mask, 'merges': fm_mask}
        if ignore_seg_ids is not None:
            masks['ignore'] = np.isin(seg, ignore_seg_ids).astype('uint32')
        ret = ret + (masks,)

    if return_ids:
        id_dict = {'splits': false_split_ids, 'merges': false_merge_ids}
        ret = ret + (id_dict,)

    return ret
