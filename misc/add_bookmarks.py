import os
import json
from mmpb.bookmarks import make_bookmark, check_bookmark
from mmpb.util import propagate_ids

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


# last three arguments are capitalized to be consistent with the keys in bookmarks dict
def add_bookmark(version, name, Position=None, Layers=None, View=None):
    bookmark_file = os.path.join(ROOT, version, 'misc', 'bookmarks.json')

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


def add_fig2_bookmarks():
    version = '0.6.6'

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
    eids = propagate_ids(ROOT, src_epi, version, cell_name, eids)
    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': eids,
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # TODO need to check that ids are propagated correctly
    check_bookmark(ROOT, version, name, 1)
    quit()

    # add bookmark for figure 2, panel C
    name = 'Figure 2C: Muscle segmentation'
    position = [112.4385016688483, 154.89179764379648, 108.0387320192992]
    view = [162.5205891508259, 0.0, 0.0, -17292.571534457347,
            0.0, 162.5205891508259, 0.0, -24390.10620770031,
            0.0, 0.0, 162.5205891508259, -17558.518378884706]

    mids = [1350, 5312, 5525, 5720, 6474, 6962, 7386,
            8143, 8144, 8177, 8178, 8885, 10027, 11092]
    src_muscle = '0.3.1'
    mids = propagate_ids(ROOT, src_muscle, version, cell_name, mids)
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
    nids = propagate_ids(ROOT, src_neph_cells, version, cell_name, nids)

    src_neph_cilia = '0.5.3'
    # TODO need to load cilia ids from file
    cids = []
    cids = propagate_ids(ROOT, src_neph_cilia, version, cilia_name, cids)

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
