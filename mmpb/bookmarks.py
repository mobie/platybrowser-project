import os
import json
import pandas as pd

ROOT_FOLDER = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
LAYER_KEYS = {'Color', 'MinValue', 'MaxValue',
              'SelectedIds', 'Tables'}
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

    if 'Tables' in keys:
        table_folder = os.path.join(ROOT_FOLDER, version, image_dict[name]['TableFolder'])
        return validate_tables(layer['Tables'], table_folder)

    return True


def make_bookmark(position, layers, view=None):
    # validate and add position
    assert isinstance(position, (list, tuple))
    assert len(position) == 3
    assert all(isinstance(pos, float) for pos in position)
    bookmark = {'Position': position}

    # validate and add layers if given
    assert isinstance(layers, dict), type(layers)
    assert all(validate_layer(version, name, layer) for name, layer in layers.items())
    bookmark.update({'Layers': layers})

    # validate and add the view if given
    if view is not None:
        assert isinstance(view, (list, tuple))
        assert len(view) == 12
        assert all(isinstance(pos, float) for pos in view)
        bookmark.update({'View': view})
    return bookmark


#
def add_bookmark(version, name, position, layers, view=None):
    bookmark_file = os.path.join(ROOT_FOLDER, version, 'misc', 'bookmarks.json')

    if os.path.exists(bookmark_file):
        with open(bookmark_file, 'r') as f:
            bookmarks = json.load(f)
    else:
        bookmarks = {}

    if name in bookmarks:
        print("Overriding bookmark for name", name)

    bookmark = make_bookmark(position, layers, view)
    bookmarks[name] = bookmark

    with open(bookmark_file, 'w') as f:
        json.dump(bookmarks, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    version = '0.6.5'

    # add the left eye bookmark
    name = 'Left eye'
    position = [177.0, 218.0, 67.0]
    layers = {'sbem-6dpf-1-whole-raw': {}}
    add_bookmark(version, name, position, layers, view=None)

    # add some random bookmark with prospr in order to set all options
    name = 'Prospr '
    position = [160.4116010078834, 87.28829031155679, 179.20095923512685]
    view = [3.1617001692017697, 17.658482138238263, 3.1136668393299125,
            -2187.524185739059, 0.0, 3.1617001692017697, -17.930892688676224,
            3194.25376750552, -17.930892688676224, 3.1136668393299125,
            0.5490234727111127, 2506.1510157337752]
    layers = {'sbem-6dpf-1-whole-raw': {},
              'prospr-6dpf-1-whole-AChE-MED': {'Color': 'Yellow',
                                               'MinValue': 0, 'MaxValue': 700}}
    add_bookmark(version, name, position, layers, view=view)

    # add bookmark for figure 2, panel
    name = 'Figure 2A - Muscles'
    position = [112.4385016688483, 154.89179764379648, 108.0387320192992]
    view = [162.5205891508259, 0.0, 0.0, -17292.571534457347,
            0.0, 162.5205891508259, 0.0, -24390.10620770031,
            0.0, 0.0, 162.5205891508259, -17558.518378884706]
    mids = [1350, 5312, 5525, 5720, 6474, 6962, 7386,
            8143, 8144, 8177, 8178, 8885, 10027, 11092]
    layers = {'sbem-6dpf-1-whole-raw': {},
              'sbem-6dpf-1-whole-segmented-cells-labels': {'SelectedIds': mids}}
    add_bookmark(version, name, position, layers, view=view)
