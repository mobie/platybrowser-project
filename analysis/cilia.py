import json
from concurrent import futures

import numpy as np
import h5py
import pandas as pd

from heimdall import view, to_source
from elf.skeleton import skeletonize
from scripts.attributes.cilia_attributes import (compute_centerline,
                                                 get_bb, load_seg,
                                                 make_indexable)


# NOTE the current paths don't look that great.
# probably need to play with the teasar parameters a bit to improve this
def view_centerline(raw, obj, path, compare_skeleton=False):
    path = make_indexable(path)
    cline = np.zeros(obj.shape, dtype='uint32')
    cline[path] = 1

    if compare_skeleton:
        coords, _ = skeletonize(obj)
        coords = make_indexable(coords)
        skel = np.zeros(obj.shape, dtype='uint32')
        skel[coords] = 1

        view(raw, obj.astype('uint32'), cline, skel)
    else:
        view(raw, obj.astype('uint32'), cline)


def check_lens(cilia_ids=None, compare_skeleton=False):
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    path_raw = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    with open('precomputed_cilia.json') as f:
        skeletons = json.load(f)

    if cilia_ids is None:
        cilia_ids = range(len(table))

    resolution = [.025, .01, .01]
    with h5py.File(path, 'r') as f, h5py.File(path_raw, 'r') as fr:
        ds = f['t00000/s00/0/cells']
        dsr = fr['t00000/s00/0/cells']

        for cid in cilia_ids:
            if cid in (0, 1, 2):
                continue

            print(cid)
            obj_path = skeletons[cid]
            if obj_path is None:
                print("Skipping cilia", cid)
                continue
            print(len(obj_path))

            bb = get_bb(table, cid, resolution)
            raw = dsr[bb]
            obj = ds[bb] == cid
            view_centerline(raw, obj, obj_path, compare_skeleton)


def precompute():
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    resolution = [.025, .01, .01]
    with h5py.File(path) as f:
        ds = f['t00000/s00/0/cells']

        def precomp(cid):
            if cid in (0, 1, 2):
                return
            print(cid)
            obj = load_seg(ds, table, cid, resolution)
            if obj.sum() == 0:
                return
            path = compute_centerline(obj, [res * 1000 for res in resolution])
            return path

        n_cilia = len(table)
        with futures.ThreadPoolExecutor(16) as tp:
            tasks = [tp.submit(precomp, cid) for cid in range(n_cilia)]
            # tasks = [tp.submit(precomp, cid) for cid in (3, 4, 5)]
            results = [t.result() for t in tasks]

        with open('precomputed_cilia.json', 'w') as f:
            json.dump(results, f)


def grid_search():
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    label_id = 11

    penalty_scales = [1000, 2500, 5000, 10000]
    penalty_exponents = [2, 4, 8, 16]

    resolution = [.025, .01, .01]
    with h5py.File(path) as f:
        ds = f['t00000/s00/0/cells']

        def precomp(cid, penalty_scale, penalty_exponent):
            print("scale:", penalty_scale, "exponent:", penalty_exponent)
            obj = load_seg(ds, table, cid, resolution)
            path = compute_centerline(obj, [res * 1000 for res in resolution],
                                      penalty_scale=penalty_scale, penalty_exponent=penalty_exponent)
            return {'penalty_scale': penalty_scale, 'penalty_exponent': penalty_exponent, 'path': path}

        with futures.ThreadPoolExecutor(16) as tp:
            tasks = [tp.submit(precomp, label_id, penalty_scale, penalty_exponent)
                     for penalty_scale in penalty_scales for penalty_exponent in penalty_exponents]
            results = [t.result() for t in tasks]

        with open('grid_search.json', 'w') as f:
            json.dump(results, f)


def eval_gridsearch():
    with open('grid_search.json') as f:
        results = json.load(f)

    path_raw = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    label_id = 11

    resolution = [.025, .01, .01]
    with h5py.File(path, 'r') as f, h5py.File(path_raw, 'r') as fr:
        ds = f['t00000/s00/0/cells']
        dsr = fr['t00000/s00/0/cells']
        bb = get_bb(table, label_id, resolution)

        raw = dsr[bb]
        obj = (ds[bb] == label_id).astype('uint32')

        sources = [to_source(raw, name='raw'), to_source(obj, name='mask')]
        for res in results:
            line = np.zeros_like(obj)
            path = make_indexable(res['path'])
            line[path] = 1
            name = '%i_%i' % (res['penalty_scale'], res['penalty_exponent'])
            sources.append(to_source(line, name=name))

        view(*sources)


if __name__ == '__main__':
    # precompute()
    # grid_search()

    check_lens([11], compare_skeleton=True)
    # eval_gridsearch()
