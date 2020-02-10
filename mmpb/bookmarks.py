import os
import json
import numpy as np
import pandas as pd

from elf.io import open_file
from pybdv.metadata import get_resolution, get_data_path
from pybdv.util import get_key
from mmpb.util import propagate_ids

ROOT_FOLDER = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
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


def validate_layer(version, name, layer):
    # check that the corresponding name exists
    image_dict = os.path.join(ROOT_FOLDER, version, 'images', 'images.json')
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
        table_folder = os.path.join(ROOT_FOLDER, version, image_dict[name]['TableFolder'])
        return validate_tables(layer['Tables'], table_folder)

    return True


# arguments are capitalized to be consistent with the keys in bookmarks dict
def make_bookmark(Position=None, Layers=None, View=None):
    # validate and add position
    if Position is not None:
        assert isinstance(Position, (list, tuple))
        assert len(Position) == 3
        assert all(isinstance(pos, float) for pos in Position)
        bookmark = {'Position': Position}

    # validate and add Layers if given
    if Layers is not None:
        assert isinstance(Layers, dict), type(Layers)
        assert all(validate_layer(version, name, layer) for name, layer in Layers.items())
        bookmark.update({'Layers': Layers})

    # validate and add the View if given
    if View is not None:
        assert isinstance(View, (list, tuple))
        assert len(View) == 12
        assert all(isinstance(pos, float) for pos in View)
        bookmark.update({'View': View})
    return bookmark


# last three arguments are capitalized to be consistent with the keys in bookmarks dict
def add_bookmark(version, name, Position=None, Layers=None, View=None):
    bookmark_file = os.path.join(ROOT_FOLDER, version, 'misc', 'bookmarks.json')

    if os.path.exists(bookmark_file):
        with open(bookmark_file, 'r') as f:
            bookmarks = json.load(f)
    else:
        bookmarks = {}

    if name in bookmarks:
        print("Overriding bookmark for name", name)

    bookmark = make_bookmark(Position, Layers, View)
    bookmarks[name] = bookmark

    with open(bookmark_file, 'w') as f:
        json.dump(bookmarks, f, indent=2, sort_keys=True)


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
    # TODO use new elf fancyness to transform the view :)
    if 'View' in bookmark:
        pass

    raw_name = 'sbem-6dpf-1-whole-raw'
    image_folder = os.path.join(root, version, 'images', 'local')
    # we always load raw
    xml_raw = os.path.join(image_folder, raw_name + '.xml')

    # make bounding box
    resolution = get_resolution(xml_raw, setup_id=0)
    resolution = scale_raw_resolution(resolution, raw_scale)
    position = [pos / res for pos, res in zip(position, resolution)]
    bb = tuple(slice(int(pos - ha), int(pos + ha)) for pos, ha in zip(position, halo))

    # load raw
    raw_path = get_data_path(xml_raw, return_absolute_path=True)
    is_h5 = os.path.splitext(raw_path)[1] == '.h5'
    raw_key = get_key(is_h5, time_point=0, setup_id=0, scale=raw_scale)
    with open_file(raw_path, 'r') as f:
        ds = f[raw_key]
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
                layer = ds[bb]
            data.append(to_source(layer, name=layer_name))

            # add mask with selected ids if given
            if 'SelectedLabelIds' in props:
                selected_ids = props['SelectedLabelIds']
                # TODO make mask for selected ids
                selected_mask = np.isin(layer, selected_ids)
                data.append(to_source(selected_mask.astype('uint32'), name='%s-selected' % layer_name))

    view(*data)


# TODO changes requested by tischi in https://github.com/platybrowser/platybrowser-backend/issues/6
# - ShowSelectedSegmentsIn3D. ShowImageIn3d
# - SelectedLabelIds
if __name__ == '__main__':
    version = '0.6.6'
    root = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'

    # add the left eye bookmark
    name = 'Left eye'
    position = [177.0, 218.0, 67.0]
    add_bookmark(version, name, position, Layers=None, View=None)

    cell_name = 'sbem-6dpf-1-whole-segmented-cells'
    cilia_name = 'sbem-6dpf-1-whole-segmented-cilia'

    # add bookmark for figure 2,panel b
    name = 'Figure 2B: Epithelial cell segmentation'
    position = [123.52869410491485, 149.1222916293258, 54.60245703388086]
    view = [36.55960993152054, -74.95830868923713, 0.0, 7198.793896571635,
            74.95830868923713, 36.55960993152054, 0.0, -14710.354798757155,
            0.0, 0.0, 83.39875970238346, -4553.7771933283475]

    src_epi = '0.5.5'
    eids = [4136, 4645, 4628, 3981, 2958, 3108, 4298]
    # TODO need to fix label id propagation
    # eids = propagate_ids(root, src_epi, version, cell_name, eids)
    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': eids,
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # TODO need to check that ids are propagated correctly
    # check_bookmark(ROOT_FOLDER, version, name, 1)
    # quit()

    # add bookmark for figure 2, panel C
    name = 'Figure 2C: Muscle segmentation'
    position = [112.4385016688483, 154.89179764379648, 108.0387320192992]
    view = [162.5205891508259, 0.0, 0.0, -17292.571534457347,
            0.0, 162.5205891508259, 0.0, -24390.10620770031,
            0.0, 0.0, 162.5205891508259, -17558.518378884706]

    mids = [1350, 5312, 5525, 5720, 6474, 6962, 7386,
            8143, 8144, 8177, 8178, 8885, 10027, 11092]
    src_muscle = '0.3.1'
    # mids = propagate_ids(root, src_muscle, version, cell_name, mids)
    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': mids,
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # add bookmark for figure 2, panel d
    name = 'Figure 2D: Nephridia segmentation'
    position = [83.30399191428275, 134.014171679122, 196.13224525293464]
    view = [49.66422153411607, 111.54766791295017, 0.0, -19052.196227198943,
            -111.54766791295017, 49.66422153411607, 0.0, 3678.656514894519,
            0.0, 0.0, 122.10412408025985, -23948.556010504282]

    src_neph_cells = '0.3.1'
    nids = [22925, 22181, 22925, 22182, 22515, 22700, 22699, 24024, 25520, 22370]
    # nids = propagate_ids(root, src_neph_cells, version, cell_name, nids)

    src_neph_cilia = '0.5.3'
    cids = []
    # TODO need to load cilia ids from file
    # cids = propagate_ids(root, src_neph_cilia, version, cilia_name, cids)

    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': nids,
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True},
              cilia_name: {'SelectedLabelIds': cids,
                           'MinValue': 0,
                           'MaxValue': 1000,
                           'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)
