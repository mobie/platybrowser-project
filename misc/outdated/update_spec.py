import json
import os
from copy import deepcopy
from glob import glob

import pandas as pd

from pybdv.metadata import write_name
from mobie.migration.migrate_v2 import migrate_project
from mobie.migration.migrate_v2.migrate_dataset import migrate_table
from mobie.metadata import read_dataset_metadata, write_dataset_metadata
from mobie.validation.utils import validate_with_schema

PROSPR_MASKS = [
    "allglands",
    "crypticsegment",
    "foregut",
    "glands",
    "head",
    "lateralectoderm",
    "prospr6-ref",
    "pygidium",
    "restofanimal",
    "vnc"
]


def parse_menu_name(source_type, source_name):
    menu_name = source_name.split('-')[0]

    if menu_name == 'prospr' and source_name.split('-')[-1] in PROSPR_MASKS:
        menu_name += '-mask'

    if source_type == 'segmentation':
        menu_name += '-segmentation'
    return menu_name


def parse_source_name(source_name):
    return source_name.split('-')[-1]


def dry_run():
    with open('./data/1.0.1/images/images.json', 'r') as f:
        sources = json.load(f)
    for name, source in sources.items():
        source_type = source['type']
        print(name, ":")
        print("name:", parse_source_name(name))
        print("menu:", parse_menu_name(source_type, name))
        print()


def check_links():
    with open('./data/datasets.json') as f:
        dss = json.load(f)['datasets']
    for ds in dss:
        table_root = os.path.join('data', ds, 'tables')
        table_folders = os.listdir(table_root)
        for tab_folder in table_folders:
            table_folder = os.path.join(table_root, tab_folder)
            table_paths = glob(os.path.join(table_root, tab_folder, '*.csv'))
            for tab in table_paths:
                if os.path.islink(tab):
                    link = os.readlink(tab)
                    abs_link = os.path.abspath(os.path.join(table_folder, link))
                    if not os.path.exists(abs_link):
                        raise RuntimeError(abs_link)


def update_raw_data():
    folder = './data/rawdata'
    xmls = glob(os.path.join(folder, '*.xml'))
    for xml in xmls:
        source_name = os.path.splitext(os.path.split(xml)[1])[0]
        source_name = parse_source_name(source_name)
        write_name(xml, 0, source_name)

    table_folders = glob(os.path.join(folder, 'tables', '*'))
    for table_folder in table_folders:
        table_paths = glob(os.path.join(table_folder, "*.csv"))
        for table_path in table_paths:
            migrate_table(table_path)


def fix_cilia_tables():
    with open('./data/datasets.json') as f:
        dss = json.load(f)['datasets']
    for ds in dss:
        table_path = os.path.join('data', ds, 'tables', 'sbem-6dpf-1-whole-segmented-cilia',
                                  'cell_id_mapping.csv')
        if (not os.path.exists(table_path)) or os.path.islink(table_path):
            continue

        table = pd.read_csv(table_path, sep='\t')
        table = table.rename(columns={'cilia_id': 'label_id'})
        table.to_csv(table_path, sep='\t', index=False)


def fix_vc_table():
    tab_path = './data/1.0.0/tables/sbem-6dpf-1-whole-segmented-cells/david_assigned_vcs.csv'
    tab = pd.read_csv(tab_path)
    tab.to_csv(tab_path, sep='\t', index=False)


def fix_bookmarks():
    ds_folder = './data/1.0.1'
    metadata = read_dataset_metadata(ds_folder)
    views = metadata['views']

    views_new = {}
    for name, view in views.items():
        if 'sourceDisplays' in view:
            new_view = deepcopy(view)

            new_displays = []
            for display in view['sourceDisplays']:
                new_display = deepcopy(display)
                if list(display.keys())[0] == 'segmentationDisplay':
                    seg_display = deepcopy(display['segmentationDisplay'])
                    seg_display['opacity'] = 0.5
                    if 'tables' in seg_display:
                        tables = seg_display['tables']
                        tables = [tab + '.tsv' for tab in tables]
                        seg_display['tables'] = tables
                    new_display['segmentationDisplay'] = seg_display
                new_displays.append(new_display)

            new_view['sourceDisplays'] = new_displays
            view = new_view
        views_new[name] = view

    metadata['views'] = views_new
    validate_with_schema(metadata, 'dataset')
    write_dataset_metadata(ds_folder, metadata)


if __name__ == '__main__':
    # dry_run()
    # check_links()

    # update_raw_data()
    # fix_cilia_tables()
    # fix_vc_table()

    # migrate_project('./data', parse_menu_name=parse_menu_name,
    #                 parse_source_name=parse_source_name)

    fix_bookmarks()
