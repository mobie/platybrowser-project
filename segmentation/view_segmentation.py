import os
import z5py
from heimdall import view, to_source
from heimdall.source_wrappers import ResizeWrapper
from pybdv.metadata import get_data_path
from pybdv.util import get_key

ROOT = '../data'


# TODO lazy loading
# TODO try auto-matching the seg scales
def view_segmentations(version, raw_scale, seg_names=[], seg_scales=[]):
    folder = os.path.join(ROOT, version, 'images', 'local')
    raw_file = os.path.join(folder, 'sbem-6dpf-1-whole-raw.xml')
    raw_file = get_data_path(raw_file, return_absolute_path=True)
    raw_key = get_key(False, time_point=0, setup_id=0, scale=raw_scale)

    with z5py.File(raw_file, 'r') as f:
        ds = f[raw_key]
        ds.n_threads = 16
        raw = ds[:]
        ref_shape = raw.shape

    data = [to_source(raw, name='raw')]

    for seg_name, seg_scale in zip(seg_names, seg_scales):
        seg_file = os.path.join(folder, seg_name + '.xml')
        seg_file = get_data_path(seg_file, return_absolute_path=True)
        seg_key = get_key(False, time_point=0, setup_id=0, scale=seg_scale)
        with z5py.File(seg_file, 'r') as f:
            ds = f[seg_key]
            ds.n_threads = 16
            seg = ds[:].astype('uint32')
            if seg.shape != ref_shape:
                # TODO !
                continue
                seg = ResizeWrapper(seg, ref_shape)
        data.append(to_source(seg, name=seg_name))

    view(*data)


def view_ganglia():
    version = '0.6.6'
    raw_scale = 4
    seg_names = ['sbem-6dpf-1-whole-segmented-ganglia']
    seg_scales = [0]
    view_segmentations(version, raw_scale, seg_names, seg_scales)


def view_cilia():
    version = '0.6.6'
    raw_scale = 4
    seg_names = ['sbem-6dpf-1-whole-segmented-cilia', 'sbem-6dpf-1-whole-segmented-nephridia']
    seg_scales = [4, 0]
    view_segmentations(version, raw_scale, seg_names, seg_scales)


if __name__ == '__main__':
    # view_ganglia()
    view_cilia()
