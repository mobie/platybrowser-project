import argparse
import os
import h5py
import imageio

from ..files.xml_utils import get_h5_path_from_xml


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


def save_tif_stack(raw, save_file):
    try:
        imageio.volwrite(save_file, raw)
        return True
    except RuntimeError:
        return False


def save_tif_slices(raw, save_file):
    print("Could not save tif stack, saving slices to folder %s instead"
          % save_file)
    save_folder = os.path.splitext(save_file)[0]
    os.makedirs(save_folder, exist_ok=True)
    for z in range(raw.shape[0]):
        out_file = os.path.join(save_folder, "z%05i.tif" % z)
        io.imsave(out_file, raw[z])


def save_tif(raw, save_file):
    if imageio is None:
        save_tif_slices(raw, save_file)
    else:
        if not save_tif_stack(raw, save_file):
            save_tif_slices(raw, save_file)


def name_to_path(name):
    name_dict = {'raw': 'images/',
                 'cells': 'segmentations/',
                 'nuclei': 'segmentations/',
                 'cilia': 'segmentations/',
                 'chromatin': 'segmentations/'}
    assert name in name_dict, "Name must be one of %s, not %s" % (str(name_dict.keys()),
                                                                  name)
    return name_dict[name]


def make_cutout(tag, name, scale, bb_start, bb_stop):
    assert all(sta < sto for sta, sto in zip(bb_start, bb_stop))

    path = os.path.join('data', tag, name_to_path(name))
    data_resolution = read_resolution(path)
    path = get_h5_path_from_xml(path, return_absolute_path=True)
    resolution = get_res_level(scale)

    data_scale = match_scale(resolution, data_resolution)

    bb_start = [int(sta / re) for sta, re in zip(bb_start, resolution)][::-1]
    bb_stop = [int(sto / re) for sto, re in zip(bb_stop, resolution)][::-1]
    bb = tuple(slice(sta, sto) for sta, sto in zip(bb_start, bb_stop))

    key = 't00000/s00/%i/cells' % data_scale
    with h5py.File(path, 'r') as f:
        ds = f[key]
        data = ds[bb]
    return data


def parse_coordinate(coord):
    coord = coord.rstrip('\n')
    pos_start = coord.find('(') + 1
    pos_stop = coord.find(')')
    coord = coord[pos_start:pos_stop]
    coord = coord.split(',')
    coord = [float(co) for co in coord]
    assert len(coord) == 3, "Coordinate conversion failed"
    return coord
