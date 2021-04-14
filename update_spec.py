import json
import os
from glob import glob
from pybdv.metadata import write_name
from mobie.migration.migrate_v2 import migrate_project
from mobie.migration.migrate_v2.migrate_dataset import migrate_table

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


if __name__ == '__main__':
    # dry_run()
    # check_links()

    update_raw_data()

    # migrate_project('./data', parse_menu_name=parse_menu_name,
    #                 parse_source_name=parse_source_name)
