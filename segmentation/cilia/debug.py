# from tqdm import tqdm
from elf.io import open_file
from cluster_tools.utils.volume_utils import blocks_in_volume, block_to_bb


def debug_vol():
    path = '../data.n5'
    key = 'volumes/cilia/segmentation'
    f = open_file(path)
    ds = f[key]
    shape = ds.shape
    block_shape = ds.chunks

    roi_begin = [7216, 12288, 7488]
    roi_end = [8640, 19040, 11392]

    blocks, blocking = blocks_in_volume(shape, block_shape, roi_begin, roi_end, return_blocking=True)
    print("Have", len(blocks), "blocks in roi")

    # check reading all blocks
    for block_id in blocks:
        print("Check block", block_id)
        block = blocking.getBlock(block_id)
        bb = block_to_bb(block)
        d = ds[bb]
        print("Have block", block_id)

    print("All checks passsed")


if __name__ == '__main__':
    debug_vol()
