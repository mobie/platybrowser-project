import json
from concurrent import futures

import numpy as np
import h5py
import pandas as pd

from heimdall import view
from scripts.attributes.cilia_attributes import (compute_centerline,
                                                 get_bb, load_seg)


def view_centerline(raw, obj, path=None):
    cline = np.zeros(obj.shape, dtype='uint32')
    if path is None:
        path, _ = compute_centerline(obj, [25, 10, 10])
    cline[path] = 1
    view(raw, obj.astype('uint32'), cline)


def check_lens(cilia_ids=None, precomputed=None):
    path = '../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
    path_raw = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    table = '../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
    table = pd.read_csv(table, sep='\t')
    table.set_index('label_id')

    if precomputed:
        with open(precomputed) as f:
            skeletons = json.load(f)
    else:
        skeletons = None

    if cilia_ids is None:
        cilia_ids = range(len(table))

    resolution = [.025, .01, .01]
    with h5py.File(path, 'r') as f, h5py.File(path_raw, 'r') as fr:
        ds = f['t00000/s00/0/cells']
        dsr = fr['t00000/s00/0/cells']
        assert ds.shape == dsr.shape

        for cid in cilia_ids:
            if cid in (0, 1, 2):
                continue

            print(cid)
            obj_path = None if skeletons is None else skeletons[cid]

            bb = get_bb(table, cid, resolution)
            # bb = tuple(slice(b.start, min(b.stop, b.start + 256)) for b in bb)
            raw = dsr[bb]
            obj = ds[bb] == cid
            view_centerline(raw, obj, obj_path)


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
            path, _ = compute_centerline(obj, [res * 1000 for res in resolution])
            return path

        n_cilia = len(table)
        with futures.ThreadPoolExecutor(16) as tp:
            tasks = [tp.submit(precomp, cid) for cid in range(n_cilia)]
            results = [t.result() for t in tasks]

        with open('precomputed_cilia.json', 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    # precompute()
    check_lens([11], compare_skeleton=True)
