import napari
import z5py


def view_intermediates():
    raw_path = '../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    raw_key = 'setup0/timepoint0/s2'

    path = '../data.n5'
    fg_key = 'volumes/cuticle/foreground'
    aff_key = 'volumes/cuticle/affinities'

    fr = z5py.File(raw_path)
    dsr = fr[raw_key]
    dsr.n_threads = 8

    f = z5py.File(path)
    ds = f[fg_key]
    ds.n_threads = 8

    dsa = f[aff_key]
    dsa.n_threads = 8

    center = [86.2, 124.8, 182.6][::-1]
    resolution = [0.05, 0.04, 0.04]
    center = [int(ce / re) for ce, re in zip(center, resolution)]
    halo = [50, 512, 512]
    bb = tuple(slice(int(ce - ha), int(ce + ha))
               for ce, ha in zip(center, halo))

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
