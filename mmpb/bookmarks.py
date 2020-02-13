import os
import json
import numpy as np
import pandas as pd

from elf.io import open_file
from elf.transformation import bdv_trafo_to_affine_matrix
from elf.wrapper.affine_volume import AffineVolume
from pybdv.metadata import get_resolution, get_data_path
from pybdv.util import get_key

LAYER_KEYS = {'Color', 'MinValue', 'MaxValue',
              'SelectedLabelIds', 'ShowImageIn3d', 'ShowSelectedSegmentsIn3d',
              'Tables'}
# TODO add all color maps supported by platybrowser
COLORMAPS = {'Glasbey', 'Viridis'}


def validate_tables(table_dict, table_folder):
    n_color_by = 0
    for table_name, table_values in table_dict.items():
        table_file = os.path.join(table_folder, table_name)
        if not os.path.exists(table_file):
            return False
        if table_values:
            if not len(table_values) == 2:
                return False
            table = pd.from_csv(table_file, sep='\t')
            col, cmap = table_values

            if col not in table.comlumns:
                return False

            if cmap not in COLORMAPS:
                return False

            n_color_by += 1

    # can color by maximally 1 column
    if n_color_by > 1:
        return False

    return True


def validate_layer(folder, name, layer):
    # check that the corresponding name exists
    image_dict = os.path.join(folder, 'images', 'images.json')
    with open(image_dict) as f:
        image_dict = json.load(f)

    if name not in image_dict:
        return False

    if not isinstance(layer, dict):
        return False

    keys = set(layer.keys())
    if len(keys - LAYER_KEYS) > 0:
        return False

    if 'ShowImageIn3d' in keys:
        show_in_3d = layer["ShowImageIn3d"]
        if not isinstance(show_in_3d, bool):
            return False

    if 'ShowSelectedSegmentsIn3d' in keys:
        show_in_3d = layer["ShowSelectedSegmentsIn3d"]
        if not isinstance(show_in_3d, bool):
            return False

    if 'Tables' in keys:
        table_folder = os.path.join(folder, image_dict[name]['TableFolder'])
        return validate_tables(layer['Tables'], table_folder)

    return True


# arguments are capitalized to be consistent with the keys in bookmarks dict
def make_bookmark(folder, Position=None, Layers=None, View=None):
    # validate and add position
    if Position is not None:
        assert isinstance(Position, (list, tuple)), type(Position)
        assert len(Position) == 3
        assert all(isinstance(pos, float) for pos in Position)
        bookmark = {'Position': Position}

    # validate and add Layers if given
    if Layers is not None:
        assert isinstance(Layers, dict), type(Layers)
        assert all(validate_layer(folder, name, layer) for name, layer in Layers.items())
        bookmark.update({'Layers': Layers})

    # validate and add the View if given
    if View is not None:
        assert isinstance(View, (list, tuple))
        assert len(View) == 12
        assert all(isinstance(pos, float) for pos in View)
        bookmark.update({'View': View})
    return bookmark


def update_bookmarks(folder, bookmarks):
    bookmark_path = os.path.join(folder, 'misc', 'bookmarks.json')
    with open(bookmark_path) as f:
        bookmark_dict = json.load(f)
    for name, bookmark in bookmarks.items():
        new_bookmark = make_bookmark(**bookmark)
        bookmark_dict[name] = new_bookmark
    with open(bookmark_path, 'w') as f:
        json.dump(bookmark_dict, f)


def scale_raw_resolution(resolution, scale):
    if scale == 0:
        return resolution
    resolution = [resolution[0],
                  resolution[1] * 2,
                  resolution[2] * 2]
    resolution = [re * 2 ** (scale - 1) for re in resolution]
    return resolution


def check_bookmark(root, version, name,
                   raw_scale, halo=[50, 512, 512],
                   layer_scale_dict={}):
    from heimdall import view, to_source
    bookmark_path = os.path.join(root, version, 'misc', 'bookmarks.json')
    with open(bookmark_path) as f:
        bookmarks = json.load(f)
    bookmark = bookmarks[name]
    assert 'Position' in bookmark, "Can only load bookmark with position"
    position = bookmark['Position']
    print("Viewing bookmark", name, "@", position)

    # use new elf fancyness to transform the view :)
    if 'View' in bookmark:
        affine_matrix = bdv_trafo_to_affine_matrix(bookmark['View'])
        # print("Found view corresponding to affine matrix:")
        # print(affine_matrix)
        # FIXME this does not work yet, probably because we need to
        # define the output coordinate system differently. Need to discuss with Tisch.
        affine_matrix = None
    else:
        affine_matrix = None

    raw_name = 'sbem-6dpf-1-whole-raw'
    image_folder = os.path.join(root, version, 'images', 'local')
    # we always load raw
    xml_raw = os.path.join(image_folder, raw_name + '.xml')

    # make bounding box
    resolution = get_resolution(xml_raw, setup_id=0)
    resolution = scale_raw_resolution(resolution, raw_scale)
    position = [pos / res for pos, res in zip(position, resolution)]
    bb = tuple(slice(int(pos - ha), int(pos + ha)) for pos, ha in zip(position, halo))
    print("Pixel position and bounding box:", position, bb)

    # load raw
    raw_path = get_data_path(xml_raw, return_absolute_path=True)
    is_h5 = os.path.splitext(raw_path)[1] == '.h5'
    raw_key = get_key(is_h5, time_point=0, setup_id=0, scale=raw_scale)
    with open_file(raw_path, 'r') as f:
        ds = f[raw_key]
        if affine_matrix is not None:
            # TODO higher order + anti-aliasing once elf supports it
            ds = AffineVolume(ds, affine_matrix=affine_matrix, order=1)
        ref_shape = ds.shape
        raw = ds[bb]

    data = [to_source(raw, name='raw')]

    if 'Layers' in bookmark:
        for layer_name, props in bookmark['Layers'].items():
            if layer_name == raw_name:
                continue
            layer_scale = layer_scale_dict.get(layer_name, 0)
            xml_layer = os.path.join(image_folder, layer_name + '.xml')
            # load layer
            layer_path = get_data_path(xml_layer, return_absolute_path=True)
            is_h5 = os.path.splitext(layer_path)[1] == '.h5'
            layer_key = get_key(is_h5, time_point=0, setup_id=0, scale=layer_scale)
            with open_file(layer_path, 'r') as f:
                ds = f[layer_key]
                shape = ds.shape
                if shape != ref_shape:
                    raise RuntimeError("Shape for layer %s = %s does not match the raw scale %s" % (layer_name,
                                                                                                    str(shape),
                                                                                                    str(ref_shape)))
                if affine_matrix is not None:
                    ds = AffineVolume(ds, affine_matrix=affine_matrix, order=0)
                layer = ds[bb]
                # int16/uint16 files are usually segmentations
                if layer.dtype in (np.dtype('int16'), np.dtype('uint16')):
                    layer = layer.astype('uint32')
            data.append(to_source(layer, name=layer_name))

            # add mask with selected ids if given
            if 'SelectedLabelIds' in props:
                selected_ids = props['SelectedLabelIds']
                # make mask for selected ids
                selected_mask = np.isin(layer, selected_ids)
                data.append(to_source(selected_mask.astype('uint32'), name='%s-selected' % layer_name))

    view(*data)
