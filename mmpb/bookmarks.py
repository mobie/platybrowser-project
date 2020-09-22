import os
import json
import numpy as np
import pandas as pd

from elf.io import open_file
from pybdv.metadata import get_resolution, get_data_path
from pybdv.util import get_key

from .format_validation import BOOKMARK_DICT_KEY
from .util import propagate_lut


def validate_tables(tables, table_folder, color_col=None):
    has_col = False

    # need also check for the column id in the default table
    if color_col is not None and 'default' not in tables:
        tables.append('default')

    for table_name in tables:

        table_file = os.path.join(table_folder, table_name + '.csv')
        assert os.path.exists(table_file), "Can't find table %s" % table_file

        if color_col is not None:
            table = pd.read_csv(table_file, sep='\t')
            if color_col in table.columns:
                has_col = True

    if color_col is not None:
        assert has_col, "Can't find color by column %s" % color_col

    # default table is implicit and we don't store it here
    if 'default' in tables:
        tables.remove('default')


def validate_layer(folder, name, layer):
    # check that the corresponding name exists
    image_dict = os.path.join(folder, 'images', 'images.json')
    with open(image_dict) as f:
        image_dict = json.load(f)

    assert isinstance(layer, dict), "Invalid layer type %s" % type(layer)
    assert name in image_dict, "Invalid layer name %s" % name

    keys = set(layer.keys())
    intersection = keys - BOOKMARK_DICT_KEY
    assert len(intersection) == 0, "Invalid layer fields %s" % str(intersection)

    if 'ShowImageIn3d' in keys:
        show_in_3d = layer["ShowImageIn3d"]
        assert isinstance(show_in_3d, bool)

    if 'ShowSelectedSegmentsIn3d' in keys:
        show_in_3d = layer["ShowSelectedSegmentsIn3d"]
        assert isinstance(show_in_3d, bool)

    if 'ColorByColumn' in keys:
        assert "Tables" in keys
        color_col = layer['ColorByColumn']
    else:
        color_col = None

    if 'Tables' in keys:
        table_folder = os.path.join(folder, image_dict[name]['TableFolder'])
        validate_tables(layer['Tables'], table_folder, color_col)


# arguments are capitalized to be consistent with the keys in bookmarks dict
def make_bookmark(folder, position=None, layers=None, view=None,
                  id_update_dicts=None):

    bookmark = {}

    # validate and add position
    if position is not None:
        assert isinstance(position, (list, tuple)), type(position)
        assert len(position) == 3
        assert all(isinstance(pos, float) for pos in position)
        bookmark.update({'position': position})

    # validate and add layers if given
    if layers is not None:
        assert isinstance(layers, dict), type(layers)
        for name, layer in layers.items():
            # needs to be updated to new bookmark format
            # validate_layer(folder, name, layer)
            if id_update_dicts is not None and name in id_update_dicts:
                ids = layer.get('selectedLabelIds', None)
                if ids is None:
                    continue
                print("Propagating ids from", ids)
                ids = propagate_lut(id_update_dicts[name], ids)
                print("to", ids)
                layer['selectedLabelIds'] = ids
                layers[name] = layer

        bookmark.update({'layers': layers})

    # validate and add the view if given
    if view is not None:
        assert isinstance(view, (list, tuple))
        assert len(view) == 12
        assert all(isinstance(pos, float) for pos in view)
        bookmark.update({'view': view})
    return bookmark


def add_bookmarks(folder, bookmarks, prev_version_folder=None,
                  updated_seg_names=None):
    # load the previous bookmarks
    bookmark_path = os.path.join(folder, 'misc', 'bookmarks', 'manuscript_bookmarks.json')
    with open(bookmark_path) as f:
        bookmark_dict = json.load(f)

    # load id update dicts (if given)
    if updated_seg_names is None:
        id_update_dicts = None
    else:
        assert prev_version_folder is not None
        id_update_dicts = {name: os.path.join(prev_version_folder,
                                              'misc',
                                              'new_id_lut_%s.json' % name)
                           for name in updated_seg_names}

    # update the bookmarks
    for name, bookmark in bookmarks.items():
        new_bookmark = make_bookmark(folder, id_update_dicts=id_update_dicts, **bookmark)
        bookmark_dict[name] = new_bookmark
    with open(bookmark_path, 'w') as f:
        json.dump(bookmark_dict, f, indent=2, sort_keys=True)


def update_bookmarks(folder, prev_version_folder, updated_seg_names):
    # load the previous bookmarks
    bookmark_path = os.path.join(prev_version_folder, 'misc', 'bookmarks', 'manuscript_bookmarks.json')
    with open(bookmark_path) as f:
        bookmarks = json.load(f)

    id_update_dicts = {name: os.path.join(folder,
                                          'misc',
                                          'new_id_lut_%s.json' % name)
                       for name in updated_seg_names}

    # update the bookmarks
    for name, bookmark in bookmarks.items():
        new_bookmark = make_bookmark(folder, id_update_dicts=id_update_dicts, **bookmark)
        bookmarks[name] = new_bookmark

    bookmark_path = os.path.join(folder, 'misc', 'bookmarks', 'manuscript_bookmarks.json')
    with open(bookmark_path, 'w') as f:
        json.dump(bookmarks, f, indent=2, sort_keys=True)


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
    from elf.transformation import bdv_to_native
    from elf.wrapper.affine_volume import AffineVolume

    bookmark_path = os.path.join(root, version, 'misc', 'bookmarks', 'manuscript_bookmarks.json')
    with open(bookmark_path) as f:
        bookmarks = json.load(f)
    bookmark = bookmarks[name]
    assert 'Position' in bookmark, "Can only load bookmark with position"
    position = bookmark['Position']
    print("Viewing bookmark", name, "@", position)

    # use new elf fancyness to transform the view :)
    if 'View' in bookmark:
        affine_matrix = bdv_to_native(bookmark['View'])
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
