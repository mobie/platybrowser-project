import json
from concurrent import futures
from math import ceil, floor

import numpy as np
import nifty.distributed as ndist
import vigra

from scipy.ndimage.morphology import binary_closing, binary_opening, binary_erosion
from skimage.morphology import convex_hull_image
from elf.io import open_file


#
# heuristics to find morphology outliers
#


def closing_ratio(seg, n_iters):
    seg_closed = binary_closing(seg, iterations=n_iters)
    m1 = float(seg.sum())
    m2 = float(seg_closed.sum())
    return m2 / m1


def opening_ratio(seg, n_iters):
    seg_opened = binary_opening(seg, iterations=n_iters)
    m1 = float(seg.sum())
    m2 = float(seg_opened.sum())
    return m1 / m2


def convex_hull_ratio(seg):
    seg_conv = convex_hull_image(seg)
    m1 = float(seg.sum())
    m2 = float(seg_conv.sum())
    return m2 / m1


def components_per_slice(seg, n_iters=0):
    n_components = 0
    n_slices = seg.shape[0]
    if n_slices == 0:
        return 0.

    for z in range(n_slices):
        if n_iters > 0:
            segz = binary_erosion(seg[z], iterations=n_iters).astype('uint32')
        else:
            segz = seg[z].astype('uint32')
        n_components += len(np.unique(vigra.analysis.labelImageWithBackground(segz))[1:])
    n_components /= n_slices
    return n_components


def read_bb_from_table(table_path, table_key, scale_factor):
    with open_file(table_path, 'r') as f:
        table = f[table_key][:]
    bb_starts = table[:, 5:8] / scale_factor
    bb_stops = table[:, 8:] / scale_factor
    bbs = [tuple(slice(int(floor(sta)),
                       int(ceil(sto))) for sta, sto in zip(starts, stops))
           for starts, stops in zip(bb_starts, bb_stops)]
    return bbs


def compute_ratios(seg_path, seg_key, table_path, table_key,
                   scale_factor, n_threads,
                   compute_ratio, seg_ids=None, sort=False):
    with open_file(seg_path, 'r') as f:
        ds = f[seg_key]
        bounding_boxes = read_bb_from_table(table_path, table_key, scale_factor)

        if seg_ids is None:
            seg_ids = np.arange(len(bounding_boxes))
        n_seg_ids = len(seg_ids)

        def _compute_ratio(seg_id):
            print("%i / %i" % (seg_id, n_seg_ids))
            bb = bounding_boxes[seg_id]
            seg = ds[bb]
            seg = seg == seg_id
            ratio = compute_ratio(seg)
            return ratio

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_compute_ratio, seg_id) for seg_id in seg_ids]
            ratios = np.array([t.result() for t in tasks])

    if sort:
        sorted_ids = np.argsort(ratios)[::-1]
        seg_ids = seg_ids[sorted_ids]
        ratios = ratios[sorted_ids]

    return seg_ids, ratios


#
# heuristics to find likely false merges
#

def get_ignore_ids(label_path, label_key,
                   ignore_names=['yolk', 'cuticle', 'neuropil']):
    with open_file(label_path, 'r') as f:
        ds = f[label_key]
        semantics = ds.attrs['semantics']
        labels = ds[:]
    ignore_label_ids = []
    for name, ids in semantics.items():
        if name in ignore_names:
            ignore_label_ids.extend(ids)
    ignore_ids = np.isin(labels, ignore_label_ids)
    ignore_ids = np.where(ignore_ids)[0]
    return ignore_ids


def weight_quantile_heuristic(seg_id, graph, node_labels, sizes, max_size, weights,
                              quantile=90):
    size_ratio = float(sizes[seg_id]) / max_size
    node_ids = np.where(node_labels == seg_id)[0]
    edges = graph.extractSubgraphFromNodes(node_ids, allowInvalidNodes=True)[0]
    this_weights = weights[edges]
    try:
        score = np.percentile(this_weights, quantile) * size_ratio
    except IndexError:
        print("Something went wrong", seg_id, this_weights.shape)
        score = 0.
    return score


def rank_false_merges(problem_path, graph_key, feat_key,
                      morpho_key, node_label_path, node_label_key,
                      ignore_ids, out_path_ids, out_path_scores,
                      n_threads, n_candidates, heuristic=weight_quantile_heuristic):
    g = ndist.Graph(problem_path, graph_key, n_threads)
    with open_file(problem_path, 'r') as f:
        ds = f[feat_key]
        ds.n_threads = n_threads
        probs = ds[:, 0]

        ds = f[morpho_key]
        ds.n_threads = n_threads
        sizes = ds[:, 1]

    with open_file(node_label_path, 'r') as f:
        ds = f[node_label_key]
        ds.n_threads = n_threads
        node_labels = ds[:]

    seg_ids = np.arange(len(sizes), dtype='uint64')
    seg_ids = seg_ids[np.argsort(sizes)[::-1]][:n_candidates]
    seg_ids = seg_ids[~np.isin(seg_ids, ignore_ids.tolist() + [0])]
    max_size = sizes[seg_ids].max()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(weight_quantile_heuristic, seg_id, g,
                           node_labels, sizes, max_size, probs) for seg_id in seg_ids]
        fm_scores = np.array([t.result() for t in tasks])

    # print("Id:", seg_ids[0])
    # sc = weight_quantile_heuristic(seg_ids[0], g,
    #                                node_labels, sizes, max_size, probs)
    # print("Score:", sc)
    # return

    # sort ids by score (decreasing)
    sorter = np.argsort(fm_scores)[::-1]
    seg_ids = seg_ids[sorter]
    fm_scores = fm_scores[sorter]

    with open(out_path_scores, 'w') as f:
        json.dump(fm_scores.tolist(), f)
    with open(out_path_ids, 'w') as f:
        json.dump(seg_ids.tolist(), f)
