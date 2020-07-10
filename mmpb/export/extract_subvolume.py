import os
import h5py
import imageio
import z5py
from pybdv.metadata import get_data_path


def parse_coordinate(coord):
    coord = coord.rstrip('\n')
    pos_start = coord.find('(') + 1
    pos_stop = coord.find(')')
    coord = coord[pos_start:pos_stop]
    coord = coord.split(',')
    coord = [float(co) for co in coord]
    assert len(coord) == 3, "Coordinate conversion failed"
    return coord


def get_res_level(level=None):
    res0 = [.01, .01, .025]
    res1 = [.02, .02, .025]
    resolutions = [res0] + [[re * 2 ** i for re in res1] for i in range(8)]
    if level is None:
        return resolutions
    else:
        assert level <= 6,\
            "Scale level %i is not supported, only supporting up to level 8" % level
        return resolutions[level]


# TODO figure out compression in imageio
def save_tif_stack(raw, save_file):
    try:
        imageio.volwrite(save_file, raw)
        return True
    except RuntimeError:
        print("Could not save tif stack, saving slices to folder %s instead"
              % save_file)
        save_tif_slices(raw, save_file)


def save_tif_slices(raw, save_file):
    save_folder = os.path.splitext(save_file)[0]
    os.makedirs(save_folder, exist_ok=True)
    for z in range(raw.shape[0]):
        out_file = os.path.join(save_folder, "z%05i.tif" % z)
        imageio.imwrite(out_file, raw[z])


def save_tif(raw, save_file):
    if imageio is None:
        save_tif_slices(raw, save_file)
    else:
        if not save_tif_stack(raw, save_file):
            save_tif_slices(raw, save_file)


def name_to_path(name):
    name_dict = {'raw': 'images/local/sbem-6dpf-1-whole-raw.xml',
                 'cells': 'images/local/sbem-6dpf-1-whole-segmented-cells.xml',
                 'nuclei': 'images/local/sbem-6dpf-1-whole-segmented-nuclei.xml',
                 'cilia': 'images/local/sbem-6dpf-1-whole-segmented-cilia.xml',
                 'chromatin': 'images/local/sbem-6dpf-1-whole-segmented-chromatin.xml'}
    assert name in name_dict, "Name must be one of %s, not %s" % (str(name_dict.keys()),
                                                                  name)
    return name_dict[name]


def name_to_base_scale(name):
    scale_dict = {'raw': 0,
                  'cells': 1,
                  'nuclei': 3,
                  'cilia': 0,
                  'chromatin': 1}
    return scale_dict[name]


def cutout_data(folder, name, scale, bb_start, bb_stop):
    assert all(sta < sto for sta, sto in zip(bb_start, bb_stop))

    path = os.path.join(folder, name_to_path(name))
    path = get_data_path(path, return_absolute_path=True)
    resolution = get_res_level(scale)

    base_scale = name_to_base_scale(name)
    assert base_scale <= scale, "%s does not support scale %i; minimum is %i" % (name, scale, base_scale)
    data_scale = scale - base_scale

    print("Resoluion of scale", scale, ":", resolution)
    bb_start_ = [int(sta / re) for sta, re in zip(bb_start, resolution)][::-1]
    bb_stop_ = [int(sto / re) for sto, re in zip(bb_stop, resolution)][::-1]
    bb = tuple(slice(sta, sto) for sta, sto in zip(bb_start_, bb_stop_))
    print("Extracting %s from pixel coordinates %s:%s" % (name,
                                                          str(bb_start_),
                                                          str(bb_stop_)))

    key = 'setup0/timepoint0/s%i' % data_scale
    with z5py.File(path, 'r') as f:
        ds = f[key]
        data = ds[bb]
    return data


# TODO support bdv-hdf5 as additional format
def to_format(path):
    ext = os.path.splitext(path)[1]
    if ext.lower() in ('.hdf', '.hdf5', '.h5'):
        return 'hdf5'
    elif ext.lower() in ('.n5',):
        return 'n5'
    elif ext.lower() in ('.tif', '.tiff'):
        return 'tif-stack'
    elif ext.lower() in ('.zr', '.zarr'):
        return 'zarr'
    else:
        print('Could not match', ext, 'to data format. Displaying the data instead')
        return 'view'


def save_data(data, path, save_format, name):
    if save_format == 'hdf5':
        with h5py.File(path) as f:
            f.create_dataset(name, data=data, compression='gzip')
    elif save_format in ('n5', 'zarr'):
        import z5py  # import here, because we don't want to make z5py mandatory dependency
        with z5py.File(path, use_zarr_format=save_format == 'zarr') as f:
            f.create_dataset(name, data=data, compression='gzip',
                             chunks=(64, 64, 64))
    elif save_format == 'tif-stack':
        save_tif_stack(data, path)
    elif save_format == 'tif-slices':
        save_tif_slices(data, path)
    elif save_format == 'view':
        from cremi_tools.viewer.volumina import view
        view([data])
    else:
        raise RuntimeError("Unsupported format %s" % save_format)


def make_cutout(folder, name, scale, bb_start, bb_stop, out_path, out_format=None):
    out_format = to_format(out_path) if out_format is None else out_format
    assert out_format in ('hdf5', 'n5', 'tif-stack', 'tif-slices', 'zarr', 'view'), "Invalid format:" % out_format
    data = cutout_data(folder, name, scale, bb_start, bb_stop)
    save_data(data, out_path, out_format, name)
