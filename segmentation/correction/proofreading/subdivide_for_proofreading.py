import os
import json
from concurrent import futures

import nifty.tools as nt
import numpy as np
import z5py
from mmpb.attributes.util import node_labels

WS_PATH = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
WS_KEY = 'volumes/paintera/proofread_cells_multiset'
SEG_PATH = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/1.0.1/images',
                        'local/sbem-6dpf-1-whole-segmented-cells.n5')
SEG_KEY = 'volumes/paintera/proofread_cells_multiset'
TMP_PATH = './data.n5'


def get_blocking(scale, block_shape):
    g = z5py.File(WS_PATH)[WS_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    return shape, blocking


def tentative_block_shape(scale, n_target_blocks=50):
    g = z5py.File(WS_PATH)[WS_KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape

    size = float(np.prod(shape))
    target_size = size / n_target_blocks

    block_len = int(target_size ** (1. / 3))
    block_len = block_len - (block_len % 64)
    block_shape = 3 * (block_len,)
    _, blocking = get_blocking(scale, block_shape)
    print("Block shape:", block_shape)
    print("Resulting in", blocking.numberOfBlocks, "blocks")
    return block_shape


def make_subdivision_vol(scale, block_shape):
    shape, blocking = get_blocking(scale, block_shape)
    f = z5py.File(TMP_PATH)
    out_key = 'labels_for_subdivision'
    if out_key in f:
        return blocking.numberOfBlocks

    ds = f.require_dataset(out_key, shape=shape, chunks=(64,) * 3, compression='gzip',
                           dtype='uint32')

    def _write_id(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        ds[bb] = block_id + 1

    n_threads = 8
    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(block_id) for block_id in range(n_blocks)]
        [t.result() for t in tasks]
    return n_blocks


def map_labels_to_blocks(n_blocks):
    tmp_folder = './tmp_subdivision_labels'
    block_labels = node_labels(SEG_PATH, SEG_KEY,
                               TMP_PATH, 'labels_for_subdivision', 'for_subdivision',
                               tmp_folder, target='local', max_jobs=48,
                               max_overlap=True, ignore_label=None)
    labels_to_blocks = {}
    for block_id in range(1, n_blocks + 1):
        this_labels = np.where(block_labels == block_id)[0]
        if this_labels[0] == 0:
            this_labels = this_labels[1:]
        labels_to_blocks[block_id] = this_labels.tolist()
    with open('./labels_to_blocks.json', 'w') as f:
        json.dump(labels_to_blocks, f)


def make_proofreading_project(block_id, block_labels):
    pass


# make the appropriate sub-volume and paintera project for each block
def make_proofreading_projects(n_blocks):
    with open('./labels_to_blocks.json', 'r') as f:
        labels_to_blocks = json.load(f)

    for block_id in range(1, n_blocks + 1):
        make_proofreading_project(block_id, labels_to_blocks[block_id])


def make_subdivision():
    # map label ids to blocks
    scale = 2
    block_shape = tentative_block_shape(scale)
    n_blocks = make_subdivision_vol(scale, block_shape)
    map_labels_to_blocks()

    #
    make_proofreading_projects(n_blocks)


if __name__ == '__main__':
    make_subdivision()
