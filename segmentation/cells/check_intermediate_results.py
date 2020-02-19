import napari
import z5py


def view_intermediates():
    raw_path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    raw_key = 'setup0/timepoint0/s1'

    path = '../data.n5'
    aff_key = 'volumes/cells/affinities/s1'

    fr = z5py.File(raw_path)
    dsr = fr[raw_key]
    dsr.n_threads = 8
    shape = dsr.shape

    f = z5py.File(path)
    dsa = f[aff_key]
    dsa.n_threads = 8

    halo = [50, 512, 512]
    roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))

    raw = dsr[bb]
    affs = dsa[(slice(None),) + bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(affs, name='affinities')


if __name__ == "__main__":
    view_intermediates()
