import os
import numpy as np
import z5py
import h5py
from heimdall import view, to_source
from heimdall.source_wrappers import ResizeWrapper
from pybdv.metadata import get_data_path
from pybdv.util import get_key

ROOT = '../data'


# TODO lazy loading
# TODO try auto-matching the seg scales
def view_segmentations(version, raw_scale, seg_names=[], seg_scales=[], bb=np.s_[:]):
    folder = os.path.join(ROOT, version, 'images', 'local')
    raw_file = os.path.join(folder, 'sbem-6dpf-1-whole-raw.xml')
    raw_file = get_data_path(raw_file, return_absolute_path=True)
    raw_key = get_key(False, time_point=0, setup_id=0, scale=raw_scale)

    with z5py.File(raw_file, 'r') as f:
        ds = f[raw_key]
        ds.n_threads = 16
        raw = ds[bb]
        ref_shape = raw.shape

    data = [to_source(raw, name='raw')]

    for seg_name, seg_scale in zip(seg_names, seg_scales):
        seg_file = os.path.join(folder, seg_name + '.xml')
        seg_file = get_data_path(seg_file, return_absolute_path=True)
        seg_key = get_key(False, time_point=0, setup_id=0, scale=seg_scale)
        with z5py.File(seg_file, 'r') as f:
            ds = f[seg_key]
            ds.n_threads = 16
            seg = ds[bb].astype('uint32')
            if seg.shape != ref_shape:
                # FIXME this will fail with bounding box
                print("Resize", ref_shape)
                seg = ResizeWrapper(to_source(seg, name=seg_name), ref_shape)
        data.append(to_source(seg, name=seg_name))

    view(*data)


def view_ganglia():
    version = '0.6.6'
    raw_scale = 4
    seg_names = ['sbem-6dpf-1-whole-segmented-ganglia']
    seg_scales = [0]
    view_segmentations(version, raw_scale, seg_names, seg_scales)


def get_cilia_bb(scale):
    scale_to_res = {0: [0.025, 0.01, 0.01],
                    1: [0.025, 0.02, 0.02],
                    2: [0.05, 0.04, 0.04],
                    3: [0.1, 0.08, 0.08],
                    4: [0.2, 0.16, 0.16],
                    5: [0.4, 0.32, 0.32],
                    6: [0.8, 0.64, 0.64]}
    resolution = scale_to_res[scale]

    halo = [50, 512, 512]
    position = [177.7, 84.1, 179.0][::-1]
    position = [pos / res for pos, res in zip(position, resolution)]

    bb = tuple(slice(int(pos - ha), int(pos + ha)) for pos, ha in zip(position, halo))
    return bb


def view_cilia():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scale', type=int)
    args = parser.parse_args()
    scale = args.scale

    version = '0.6.6'
    raw_scale = scale
    seg_names = ['sbem-6dpf-1-whole-segmented-cilia']
    seg_scales = [scale]

    bb = get_cilia_bb(scale)

    # seg_names = ['sbem-6dpf-1-whole-segmented-cilia', 'sbem-6dpf-1-whole-segmented-nephridia']
    # seg_scales = [4, 0]

    view_segmentations(version, raw_scale, seg_names, seg_scales, bb=bb)


def get_cell_bb(scale, halo=[50, 512, 512]):
    path = '../data/0.0.0/images/local/sbem-6dpf-1-whole-segmented-cells.h5'
    key = 't00000/s00/%i/cells' % scale
    with h5py.File(path, 'r') as f:
        ds = f[key]
        shape = ds.shape

    central = [sh // 2 for sh in shape]
    bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(central, halo))
    return bb


def view_cells():
    version = '0.6.6'
    raw_scale = 2
    seg_names = ['sbem-6dpf-1-whole-segmented-cells']
    seg_scales = [1]

    bb = get_cell_bb(seg_scales[0])

    view_segmentations(version, raw_scale, seg_names, seg_scales, bb=bb)


if __name__ == '__main__':
    # view_ganglia()
    # view_cilia()
    view_cells()
