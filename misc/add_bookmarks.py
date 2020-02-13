import os
import json
import numpy as np
import pandas as pd
from mmpb.bookmarks import make_bookmark

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


# last three arguments are capitalized to be consistent with the keys in bookmarks dict
def add_bookmark(version, name, Position=None, Layers=None, View=None):
    folder = os.path.join(ROOT, version)
    bookmark_file = os.path.join(folder, 'misc', 'bookmarks.json')

    if os.path.exists(bookmark_file):
        with open(bookmark_file, 'r') as f:
            bookmarks = json.load(f)
    else:
        bookmarks = {}

    if name in bookmarks:
        print("Overriding bookmark for name", name)

    bookmark = make_bookmark(folder, Position=Position, Layers=Layers, View=View)
    bookmarks[name] = bookmark

    with open(bookmark_file, 'w') as f:
        json.dump(bookmarks, f, indent=2, sort_keys=True)


def get_nephridia_ids(version, side):
    assert side in (1, 2)
    table_path = os.path.join(ROOT, version, 'tables', 'sbem-6dpf-1-whole-segmented-cells', 'regions.csv')
    region_table = pd.read_csv(table_path, sep='\t')
    label_ids = region_table['label_id'].values.astype('uint32')
    nephridia_ids = region_table['nephridia'].astype('uint32')
    return label_ids[nephridia_ids == side]


def get_cilia_ids(version, nephridia_ids):
    table_path = os.path.join(ROOT, version, 'tables', 'sbem-6dpf-1-whole-segmented-cilia', 'cell_mapping.csv')
    mapping_table = pd.read_csv(table_path, sep='\t')
    cilia_ids = mapping_table['label_id'].values.astype('uint32')
    cell_ids = mapping_table['cell_id'].values.astype('uint32')
    id_mask = np.isin(cell_ids, nephridia_ids)
    return cilia_ids[id_mask]


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

    eids = [4211, 4724, 4707, 3031, 4056, 3181, 4373]
    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': eids,
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    # add bookmark for figure 2, panel c
    name = 'Figure 2C: Muscle segmentation'
    position = [112.4385016688483, 154.89179764379648, 108.0387320192992]
    view = [162.5205891508259, 0.0, 0.0, -17292.571534457347,
            0.0, 162.5205891508259, 0.0, -24390.10620770031,
            0.0, 0.0, 162.5205891508259, -17558.518378884706]

    mids = [1425, 5385, 5598, 5795, 6552, 7044, 7468, 8264,
            8230, 8231, 8987, 9185, 10167, 11273]
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

    # nephridia ids: cell ids for left nephridia
    nids = get_nephridia_ids(version, side=2)
    # cilia ids: ids for nephridia corresponding to the left cell
    cids = get_cilia_ids(version, nids)

    layers = {'sbem-6dpf-1-whole-raw': {},
              cell_name: {'SelectedLabelIds': nids.tolist(),
                          'MinValue': 0,
                          'MaxValue': 1000,
                          'ShowSelectedSegmentsIn3d': True},
              cilia_name: {'SelectedLabelIds': cids.tolist(),
                           'MinValue': 0,
                           'MaxValue': 1000,
                           'ShowSelectedSegmentsIn3d': True}}
    add_bookmark(version, name, Position=position, Layers=layers, View=view)


def add_fig5_bookmarks():
    version = '0.6.6'
    name = "Figure 5C: Gene clustering full body"
    layers = {"sbem-6dpf-1-whole-raw": {},
              "sbem-6dpf-1-whole-segmented-cells": {"Tables": {"gene_clusters": ["clusters", "glasbey"]}}}
    position = [34.3200812765686, 282.12761791060416, 105.03347558347836]
    view = [-0.9529921731518778,
            1.1849892416847392,
            0.00814080433223175,
            376.53351985923825,
            -0.5079154826309126,
            -0.4178950423193335,
            1.3710745617220128,
            196.32270696996784,
            1.0706482326644668,
            0.8565185861315733,
            0.6576839143510288,
            -347.4711101247537]
    add_bookmark(version, name, Position=position, Layers=layers, View=view)


def add_fig6_bookmarks():
    version = '0.6.6'
    name = "Figure 6A: Assignment by overlap"
    layers = {"sbem-6dpf-1-whole-raw": {},
              "prospr-6dpf-1-whole-msx": {"Color": "Green"},
              "prospr-6dpf-1-whole-patched": {"Color": "Blue"},
              "sbem-6dpf-1-whole-segmented-cells": {
                "MaxValue": 1000,
                "MinValue": 0,
                "SelectedLabelIds": [
                  32095,
                  31961
                ],
                "ShowSelectedSegmentsIn3d": False}}
    position = [119.96805985437719,
                99.19674811104227,
                263.3359252291697]
    view = [-13.194629461107638,
            10.214861012475334,
            0.30412004228770306,
            1308.0492653634751,
            9.677834961769312,
            12.330343938455595,
            5.73029230776116,
            -3359.034760194792,
            3.2825881491691082,
            4.706734959124246,
            -15.671802135929735,
            3266.795464793777]
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    name = "Figure 6B: Assignment by overlap"
    layers = {"sbem-6dpf-1-whole-raw": {},
              "prospr-6dpf-1-whole-lhx6": {"Color": "Green"},
              "prospr-6dpf-1-whole-wnt5": {"Color": "Blue"},
              "sbem-6dpf-1-whole-segmented-cells": {
                "MaxValue": 1000,
                "MinValue": 0,
                "SelectedLabelIds": [
                  6766,
                  6913
                ],
                "ShowSelectedSegmentsIn3d": False}}
    position = [196.76744017557536,
                178.25856120112738,
                71.18235953395656]
    view = [-6.194255640646106,
            7.933757678523062,
            0.04504878697308157,
            629.360918560249,
            7.898413563957639,
            6.161063641179686,
            0.9857490646362296,
            -2224.580902902456,
            0.749402073270016,
            0.6419712954778718,
            -10.017066068703377,
            451.14359129392136]
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    name = "Figure 6F: Level of gene expression after VC assignment"
    layers = {"sbem-6dpf-1-whole-raw": {},
              "sbem-6dpf-1-whole-segmented-outside": {},
              "sbem-6dpf-1-whole-segmented-cells": {
                "Tables": {
                  "vc_assignments": ["expression_sum", "glasbey"]
                }}}
    position = [151.49356333542673,
                142.11330737746303,
                124.51951538415905]
    view = [-1.6868627317328129,
            2.5114207685721133,
            -0.040111647775085676,
            666.6372173919165,
            -1.0506500045055518,
            -0.6616293061174092,
            2.7591176716716426,
            356.629046586707,
            2.2814420415901586,
            1.5522118029711398,
            1.2409713237596134,
            -720.738885334493]
    add_bookmark(version, name, Position=position, Layers=layers, View=view)

    name = "Symmetric cell pairs"
    layers = {"sbem-6dpf-1-whole-raw": {},
              "sbem-6dpf-1-whole-segmented-outside": {},
              "sbem-6dpf-1-whole-segmented-cells": {
                "Tables": {
                    "symmetric_cells": ["pair_index", "glasbey"]
                }}}
    position = [151.49356333542673,
                142.11330737746303,
                124.51951538415905]
    view = [-1.6868627317328129,
            2.5114207685721133,
            -0.040111647775085676,
            666.6372173919165,
            -1.0506500045055518,
            -0.6616293061174092,
            2.7591176716716426,
            356.629046586707,
            2.2814420415901586,
            1.5522118029711398,
            1.2409713237596134,
            -720.738885334493]
    add_bookmark(version, name, Position=position, Layers=layers, View=view)


if __name__ == '__main__':
    # name = 'Figure 2C: Muscle segmentation'
    # check_bookmark(ROOT, '0.6.6', name, 1)

    add_fig2_bookmarks()
    # add_fig5_bookmarks()
    # add_fig6_bookmarks()
