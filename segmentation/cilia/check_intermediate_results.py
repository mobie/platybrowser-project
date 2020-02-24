import napari
import z5py


def view_intermediates():
    raw_path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    raw_key = 'setup0/timepoint0/s0'

    path = '../data.n5'
    fg_key = 'volumes/cilia/foreground'
    aff_key = 'volumes/cilia/affinities'
    seg_key = 'volumes/cilia/segmentation'

    fr = z5py.File(raw_path)
    dsr = fr[raw_key]
    dsr.n_threads = 8

    f = z5py.File(path)
    ds = f[fg_key]
    ds.n_threads = 8

    dsa = f[aff_key]
    dsa.n_threads = 8

    center = [90.5, 138.7, 196.1][::-1]
    resolution = [0.025, 0.01, 0.01]
    center = [int(ce / re) for ce, re in zip(center, resolution)]
    halo = [75, 512, 512]
    bb = tuple(slice(int(ce - ha), int(ce + ha))
               for ce, ha in zip(center, halo))

    raw = dsr[bb]
    fg = ds[bb]
    affs = dsa[(slice(None),) + bb]

    if seg_key in f:
        ds_seg = f[seg_key]
        ds_seg.n_threads = 8
        seg = ds_seg[bb]
    else:
        seg = None

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(fg, name='foreground')
        viewer.add_image(affs, name='affinities')
        if seg is not None:
            viewer.add_labels(seg, name='mws-segmentation')


if __name__ == "__main__":
    view_intermediates()
