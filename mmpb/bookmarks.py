import os
import json
import pandas as pd

ROOT_FOLDER = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
LAYER_KEYS = {'Color', 'MinValue', 'MaxValue',
              'SelectedIds', 'Tables', 'ShowIn3d'}
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

    if 'ShowIn3d' in keys:
        show_in_3d = layer["ShowIn3d"]
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


if __name__ == '__main__':
    version = '0.6.6'

    # add the left eye bookmark
    name = 'Left eye'
    position = [177.0, 218.0, 67.0]
    add_bookmark(version, name, position, Layers=None, View=None)

    # add bookmark for figure 2,panel b
    name = 'Figure 2B: Epithelial cell segmentation'
    position = [123.52869410491485, 149.1222916293258, 54.60245703388086]
    view = [36.55960993152054, -74.95830868923713, 0.0, 7198.793896571635,
            74.95830868923713, 36.55960993152054, 0.0, -14710.354798757155,
            0.0, 0.0, 83.39875970238346, -4553.7771933283475]
    # TODO ids need to be propagated
    eids = [4136, 4645, 4628, 3981, 2958, 3108, 4298]
    layers = {'sbem-6dpf-1-whole-raw': {},
              'sbem-6dpf-1-whole-segmented-cells': {'SelectedIds': eids,
                                                    'MinValue': 0,
                                                    'MaxValue': 1000,
                                                    'ShowIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # add bookmark for figure 2, panel C
    name = 'Figure 2C: Muscle segmentation'
    position = [112.4385016688483, 154.89179764379648, 108.0387320192992]
    view = [162.5205891508259, 0.0, 0.0, -17292.571534457347,
            0.0, 162.5205891508259, 0.0, -24390.10620770031,
            0.0, 0.0, 162.5205891508259, -17558.518378884706]
    # TODO ids need to be propagated
    mids = [1350, 5312, 5525, 5720, 6474, 6962, 7386,
            8143, 8144, 8177, 8178, 8885, 10027, 11092]
    layers = {'sbem-6dpf-1-whole-raw': {},
              'sbem-6dpf-1-whole-segmented-cells': {'SelectedIds': mids,
                                                    'MinValue': 0,
                                                    'MaxValue': 1000,
                                                    'ShowIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # add bookmark for figure 2, panel d
    name = 'Figure 2D: Nephridia segmentation'
    position = [83.30399191428275, 134.014171679122, 196.13224525293464]
    view = [49.66422153411607, 111.54766791295017, 0.0, -19052.196227198943,
            -111.54766791295017, 49.66422153411607, 0.0, 3678.656514894519,
            0.0, 0.0, 122.10412408025985, -23948.556010504282]
    # TODO ids need to be propagated
    nids = [22925, 22181, 22925, 22182, 22515, 22700, 22699, 24024, 25520, 22370]
    cids = []
    layers = {'sbem-6dpf-1-whole-raw': {},
              'sbem-6dpf-1-whole-segmented-cells': {'SelectedIds': nids,
                                                    'MinValue': 0,
                                                    'MaxValue': 1000,
                                                    'ShowIn3d': True},
              'sbem-6dpf-1-whole-segmented-cilia': {'SelectedIds': cids,
                                                    'MinValue': 0,
                                                    'MaxValue': 1000,
                                                    'ShowIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)
