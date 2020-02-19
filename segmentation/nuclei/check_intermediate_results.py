import napari
import z5py


def view_intermediates():
    raw_path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    raw_key = 'setup0/timepoint0/s3'

    path = '../data.n5'
    fg_key = 'volumes/nuclei/foreground'
    aff_key = 'volumes/nuclei/affinities'

    fr = z5py.File(raw_path)
    dsr = fr[raw_key]
    dsr.n_threads = 8
    shape = dsr.shape

    f = z5py.File(path)
    ds = f[fg_key]
    ds.n_threads = 8

    dsa = f[aff_key]
    dsa.n_threads = 8

    halo = [25, 256, 256]
    roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))

    raw = dsr[bb]
    fg = ds[bb]
    affs = dsa[(slice(None),) + bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(fg, name='foreground')
        viewer.add_image(affs, name='affinities')


if __name__ == "__main__":
    view_intermediates()
