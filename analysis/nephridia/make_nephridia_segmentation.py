import numpy as np
import pandas as pd
import elf.parallel
import vigra
from elf.io import open_file
from pybdv import make_bdv


# leaving this here to reproduce issues with elf.parallel.label
def make_nephridia_segmentation_depr():
    table_path = '../../data/0.6.5/tables/sbem-6dpf-1-whole-segmented-cilia-labels/cell_mapping.csv'
    seg_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    # TODO
    out_path = '/g/kreshuk/pape/Work/nephr_tmp.n5'
    out_key = 'test'

    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id'].values
    cell_ids = np.unique(cell_ids)
    if cell_ids[0] == 0:
        cell_ids = cell_ids[1:]
    print(cell_ids)

    scale = 4
    key = 't00000/s00/%i/cells' % scale
    with open_file(seg_path, 'r') as f, open_file(out_path) as f_out:
        ds = f[key]
        print(ds.shape)

        out = f_out.require_dataset(out_key, shape=ds.shape, chunks=ds.chunks, compression='gzip',
                                    dtype='uint8')

        print("Isin ...")
        out = elf.parallel.isin(ds, cell_ids, out=out, n_threads=16, verbose=True)
        print("Label ...")
        # FIXME label is still not doing a completely accurate job of merging across block boundaries
        out = elf.parallel.label(out, out, n_threads=16, verbose=True)
        print("Compute max ...")
        max_id = elf.parallel.max(out, n_threads=16, verbose=True)
        print("Max component id:", max_id)


# leaving this here to reproduce issues with elf.parallel.label
def make_nephridia_segmentation():
    table_path = '../../data/0.6.5/tables/sbem-6dpf-1-whole-segmented-cilia-labels/cell_mapping.csv'
    seg_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'

    out_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-nephridia.xml'

    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id'].values
    cell_ids = np.unique(cell_ids)
    if cell_ids[0] == 0:
        cell_ids = cell_ids[1:]
    print(cell_ids)

    scale = 4
    key = 't00000/s00/%i/cells' % scale
    with open_file(seg_path, 'r') as f:
        ds = f[key]
        seg = ds[:].astype('uint32')
        bshape = (32, 256, 256)

        tmp = np.zeros_like(seg)
        print("Isin ...")
        tmp = elf.parallel.isin(seg, cell_ids, out=tmp, n_threads=16, verbose=True, block_shape=bshape)
        print("Label ...")
        tmp = vigra.analysis.labelVolumeWithBackground(tmp)

        print("Size filter ...")
        ids, counts = elf.parallel.unique(tmp, return_counts=True, n_threads=16, verbose=True, block_shape=bshape)
        keep_ids = np.argsort(counts)[::-1]
        keep_ids = ids[keep_ids[:3]]
        assert keep_ids[0] == 0

        out = np.zeros(tmp.shape, dtype='uint8')
        for new_id, keep_id in enumerate(keep_ids[1:], 1):
            out[tmp == keep_id] = new_id

    factors = 3 * [[2, 2, 2]]
    res = [.4, .32, .32]
    make_bdv(out, out_path, factors, resolution=res, unit='micrometer')


def append_nephridia_table():
    table_path = '../../data/0.6.5/tables/sbem-6dpf-1-whole-segmented-cilia-labels/cell_mapping.csv'
    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id'].values
    cell_ids = np.unique(cell_ids)
    if cell_ids[0] == 0:
        cell_ids = cell_ids[1:]

    out_table_path = '../../data/0.6.5/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv'
    seg_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    nep_path = '../../data/0.6.5/segmentations/sbem-6dpf-1-whole-segmented-nephridia.h5'

    table = pd.read_csv(out_table_path, sep='\t')
    new_col = np.zeros(len(table), dtype='float32')

    print("Loading volumes ...")
    scale = 4
    key = 't00000/s00/%i/cells' % scale
    with open_file(seg_path, 'r') as f:
        seg = f[key][:]
    scale = 0
    key = 't00000/s00/%i/cells' % scale
    with open_file(nep_path, 'r') as f:
        nep = f[key][:]
    assert nep.shape == seg.shape

    print("Iterating over cells ...")
    for cid in cell_ids:
        nid = np.unique(nep[seg == cid])
        if 0 in nid:
            nid = nid[1:]
        assert len(nid) == 1
        new_col[cid] = nid

    table['nephridia'] = new_col
    table.to_csv(out_table_path, sep='\t', index=False)


if __name__ == '__main__':
    # make_nephridia_segmentation()
    append_nephridia_table()
