import os
import json
from . import attributes
from .export import export_segmentation
from .files import copy_tables, copy_file
from .util import read_resolution

VERSION_FILE = "data/versions.json"

IMAGE_FIELD_NAMES = {'Color', 'MaxValue', 'MinValue', 'Type'}
MASK_FIELD_NAMES = {'Color', 'MaxValue', 'MinValue', 'Type'}
SEGMENTATION_FIELD_NAMES = {'ColorMap', 'MaxValue', 'MinValue', 'Type'}


def make_new_seg_dict(paintera_project, table_update_function,
                      postprocess, map_to_background):
    new_seg_dict = {'PainteraProject': paintera_project}
    if table_update_function is not None:
        new_seg_dict['TableUpdateFunction'] = table_update_function
    if postprocess is not None:
        new_seg_dict['Postprocess'] = postprocess
    if map_to_background is not None:
        new_seg_dict['MapToBackground'] = map_to_background
    return new_seg_dict


def add_data(name, properties, folder, target, max_jobs):

    type_ = properties['Type']
    input_path = properties.pop('InputPath', None)

    # additional segmentation options
    input_table_folder = properties.pop('InputTableFolder', None)

    # options for dynamic segmentation
    map_to_background = properties.pop('MapToBackground', None)
    paintera_project = properties.pop('PainteraProject', None)
    postprocess = properties.pop('Postprocess', None)
    table_update_function = properties.pop('TableUpdateFunction', None)

    image_dict_path = os.path.join(folder, 'images', 'images.json')
    with open(image_dict_path) as f:
        image_dict = json.load(f)
    storage_path = os.path.join(folder, 'images', 'local', name + '.xml')

    if type_ == 'Image':
        if paintera_project or postprocess or table_update_function or input_table_folder:
            raise ValueError("Type: Image does not support segmentation options")
        if input_path is None:
            raise ValueError("Need InputPath")
        if set(properties.keys()) != IMAGE_FIELD_NAMES:
            raise ValueError("Invalid fields for Type: Image")
        # TODO check that all values in properties are valid

        # copy the file
        if os.path.splitext(input_path)[1] != '.xml':
            raise ValueError("Invalid input format, expected xml")
        copy_file(input_path, storage_path)

    elif type_ == 'Mask':
        if paintera_project or postprocess or table_update_function or input_table_folder:
            raise ValueError("Type: Mask does not support segmentation options")
        if input_path is None:
            raise ValueError("Need InputPath")
        if set(properties.keys()) != MASK_FIELD_NAMES:
            raise ValueError("Invalid fields for Type: Mask")
        # TODO check that all values in properties are valid

        # copy the file
        if os.path.splitext(input_path)[1] != '.xml':
            raise ValueError("Invalid input format, expected xml")
        storage_path = os.path.join(folder, 'images', 'local', name + '.xml')
        copy_file(input_path, storage_path)

    # Type = Segmentation and input_path is None -> dynamic segmentation
    elif type_ == 'Segmentation' and input_path is None:
        if paintera_project is None:
            raise ValueError("Need paintera project")

        paintera_path, paintera_key = paintera_project
        tmp_folder = 'tmp_export_%s' % name
        export_segmentation(paintera_path, paintera_key, name,
                            None, folder, storage_path, tmp_folder,
                            postprocess, map_to_background, target, max_jobs)

        # call the table update function if given
        if table_update_function is not None:
            tab_update = getattr(attributes, table_update_function, None)
            if tab_update is None:
                raise ValueError("Invalid table update function")

            out_table_folder = os.path.join(folder, 'tables', name)
            os.makedirs(out_table_folder, exist_ok=True)

            tmp_folder = 'tmp_table_%s' % name
            resolution = read_resolution(paintera_path, paintera_key)
            tab_update(None, folder, name, tmp_folder, resolution,
                       target=target, max_jobs=max_jobs, seg_has_changed=False)
            properties.update({'TableFolder': out_table_folder})

        new_seg_dict = make_new_seg_dict(paintera_project,
                                         table_update_function,
                                         postprocess,
                                         map_to_background)
        seg_dict_path = os.path.join(folder, 'misc', 'dynamic_segmentations.json')
        with open(seg_dict_path) as f:
            seg_dict = json.load(f)
        seg_dict.update({name: new_seg_dict})
        with open(seg_dict_path, 'w') as f:
            json.dump(seg_dict, f)

    # Type = Segmentation and input_path is not None -> static segmentation
    elif type_ == 'Segmentation' and input_path is not None:
        if paintera_project or postprocess or table_update_function:
            raise ValueError("Static segmentation not support dynamic segmentation options")

        if set(properties.keys()) != SEGMENTATION_FIELD_NAMES:
            raise ValueError("Invalid fields for Type: Segmentation")

        # TODO check that all values in properties are valid

        # copy the file
        if os.path.splitext(input_path)[1] != '.xml':
            raise ValueError("Invalid input format, expected xml")
        copy_file(input_path, storage_path)

        # copy tables if given
        if input_table_folder is not None:
            out_table_folder = os.path.join(folder, 'tables', name)
            copy_tables(input_table_folder, out_table_folder)
            properties.update({'TableFolder': out_table_folder})

    # update the properties and the image dict
    properties.update({'Storage': {'local': storage_path}})
    image_dict.update({name: properties})
    with open(image_dict_path, 'w') as f:
        json.dump(image_dict, f, indent=2, sort_keys=True)


def add_version(tag):
    if not os.path.exists(VERSION_FILE):
        versions = []
    with open(VERSION_FILE) as f:
        versions = json.load(f)
    versions.append(tag)
    with open(VERSION_FILE, 'w') as f:
        json.dump(versions, f)


def get_latest_version():
    with open(VERSION_FILE) as f:
        versions = json.load(f)
    return versions[-1]


def make_folder_structure(root):
    # make all sub-folders
    os.makedirs(os.path.join(root, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(root, 'images', 'local'), exist_ok=True)
    os.makedirs(os.path.join(root, 'images', 'remote'), exist_ok=True)
    os.makedirs(os.path.join(root, 'misc'), exist_ok=True)


def get_modality_names(root, version):
    """ Get names of the current data modalities.

    See https://github.com/platybrowser/platybrowser-backend#file-naming
    for the source naming conventions.
    """
    image_dict = os.path.join(root, version, 'images', 'images.json')
    with open(image_dict, 'r') as f:
        image_dict = json.load(f)
    names = list(image_dict.keys())
    names = set('-'.join(name.split('-')[:4]) for name in names)
    return list(names)
