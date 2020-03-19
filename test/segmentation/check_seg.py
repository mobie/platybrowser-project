import os
import z5py
from mmpb.export.check_segmentation import check_connected_components

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


def prepare_vol(path, key_root):
    with z5py.File(path) as f:
        g = f[key_root]
        ds0 = g['s0']
        max_id = ds0.attrs['maxId']
        for ds in g.values():
            if 'maxId' not in ds.attrs:
                ds.attrs['maxId'] = max_id


def check_cell_segmentation(version, scale=1):
    vfolder = os.path.join(ROOT, version)

    seg_name = 'images/local/sbem-6dpf-1-whole-segmented-cells.n5'
    seg_path = os.path.join(vfolder, seg_name)
    seg_key_root = 'setup0/timepoint0'
    prepare_vol(seg_path, seg_key_root)
    seg_key = os.path.join(seg_key_root, 's%i' % scale)

    ws_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    ws_key_root = 'volumes/paintera/proofread_cells_multiset/data'
    prepare_vol(ws_path, ws_key_root)
    ws_key = ws_key = os.path.join(ws_key_root, 's%i' % scale)

    target = 'slurm'
    max_jobs = 200
    passed = check_connected_components(ws_path, ws_key, seg_path, seg_key,
                                        'tmp_check_cc', target, max_jobs)

    if passed:
        print("Connected components and node labeling agree")
    else:
        print("Connected components and node labeling DO NOT agree")


if __name__ == '__main__':
    check_cell_segmentation('1.0.1', scale=0)
    # check_cell_segmentation('0.6.5', scale=0)
