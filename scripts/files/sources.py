import json
import os
from shutil import copyfile
from .checks import check_bdv, check_tables, check_paintera
from ..check_attributes import check_attributes
from .xml_utils import get_h5_path_from_xml, copy_xml_with_newpath

RAW_FOLDER = 'data/rawdata'
SOURCE_FILE = 'data/sources.json'
SEGMENTATION_FILE = 'data/segmentations.json'
IMAGE_FILE = 'data/images.json'
PRIVATE_FILE = 'data/privates.json'

# TODO we need additional functionality:
# - remove images and segmentations
# - update images and segmentations


def get_sources():
    """ Get names of the current data sources.

    See https://git.embl.de/tischer/platy-browser-tables/README.md#file-naming
    for the source naming conventions.
    """
    if not os.path.exists(SOURCE_FILE):
        return []
    with open(SOURCE_FILE) as f:
        sources = json.load(f)
    return sources


def source_to_prefix(source):
    return '%s-%s-%s-%s' % (source['modality'],
                            source['stage'],
                            source['id'],
                            source['region'])


def get_source_names():
    """ Get the name prefixes corresponding to all sources.
    """
    sources = get_sources()
    prefixes = [source_to_prefix(source) for source in sources]
    return prefixes


def get_image_names():
    if not os.path.exists(IMAGE_FILE):
        return []
    with open(IMAGE_FILE) as f:
        names = json.load(f)
    return names


def get_segmentations():
    if not os.path.exists(SEGMENTATION_FILE):
        return {}
    with open(SEGMENTATION_FILE) as f:
        segmentations = json.load(f)
    return segmentations


def get_segmentation_names():
    segmentations = get_segmentations()
    return list(segmentations.keys())


def add_source(modality, stage, id=1, region='whole'):
    """ Add a new data source

    See https://git.embl.de/tischer/platy-browser-tables/README.md#file-naming
    for the source naming conventions.
    """
    if not isinstance(modality, str):
        raise ValueError("Expected modality to be a string, not %s" % type(modality))
    if not isinstance(stage, str):
        raise ValueError("Expected stage to be a string, not %s" % type(id))
    if not isinstance(id, int):
        raise ValueError("Expected id to be an integer, not %s" % type(id))
    if not isinstance(region, str):
        raise ValueError("Expected region to be a string, not %s" % type(id))
    sources = get_sources()
    source = {'modality': modality, 'stage': stage, 'id': str(id), 'region': region}

    if source in sources:
        raise RuntimeError("Source is already present")

    sources.append(source)
    with open(SOURCE_FILE, 'w') as f:
        json.dump(sources, f)


def get_privates():
    if not os.path.exists(PRIVATE_FILE):
        return []
    with open(PRIVATE_FILE) as f:
        return json.load(f)


def add_to_privates(name):
    privates = get_privates()
    privates.append(name)
    with open(PRIVATE_FILE, 'w') as f:
        json.dump(privates, f)


def add_image(source_name, name, input_path, copy_data=True, is_private=False):
    """ Add image volume to the platy browser data.

    Parameter:
        source_name [str] - prefix of the primary data source.
        name [str] - name of the data.
        input_path [str] - path to the data that should be added.
            Data needs to be in bdv-hdf5 format and the path needs to point to the xml.
        copy_data [bool] - whether to copy the data. This should be set to True,
            unless adding an image volume that is already in the rawdata folder (default: True).
        is_private [bool] - whether this data is private (default: False).
    """
    # validate the inputs
    source_names = get_source_names()
    if source_name not in source_names:
        raise ValueError("""Source %s is not in the current sources.
                            Use 'add_source' to add a new source.""" % source_name)
    if not check_bdv(input_path):
        raise ValueError("Expect input to be in bdv format")
    output_name = '%s-%s' % (source_name, name)
    names = get_image_names()
    if output_name in names:
        raise ValueError("Name %s is already taken" % output_name)

    h5_path = get_h5_path_from_xml(input_path, return_absolute_path=True)
    name_h5 = '%s.h5' % output_name
    out_xml = os.path.join(RAW_FOLDER, '%s.xml' % output_name)
    out_h5 = os.path.join(RAW_FOLDER, name_h5)
    if copy_data:
        # copy h5 and xml to the rawdata folder, update the xml with new relative path
        copyfile(h5_path, out_h5)
        copy_xml_with_newpath(input_path, out_xml, name_h5)
    else:
        if not os.path.exists(out_xml) or not os.path.exists(out_h5):
            raise RuntimeError("""You did not specify to copy the data, but
                                  %s and %s do not exist yet""" % (out_xml, out_h5))

    # add name to the name list and serialze
    names.append(output_name)
    with open(IMAGE_FILE, 'w') as f:
        json.dump(names, f)

    # add the name to the private list if is_private == True
    if is_private:
        add_to_privates(output_name)


def add_segmentation(source_name, name, segmentation_path=None,
                     table_path_dict=None, paintera_project=None,
                     resolution=None, table_update_function=None,
                     copy_data=True, is_private=False):
    """ Add segmentation volume to the platy browser data.

    We distinguish between static and dynamic segmentations. A dynamic segmentation is generated from
    a paintera project and can change due to corrections made in paintera, while a static segmentation
    is just added once and does not change.
    In addition, we can add tables associated with the data that contain derived data.
    For static segmentations you need to pass a dict containing table names and paths,
    for dynamic segmentations you need to register a function name that will compute the tables.

    Adding a static segmentation:
    ```
    # if you add tables, one must have the name 'default'
    add_segmentation(source_name, seg_name,
                     segmentation_path='/path/to/input-segmentation.xml',
                     table_path_dict={'default': '/path/to/default-table.csv',
                                      'other': '/path/to/other-table.csv'})
    ```

    Adding a dynamic segmentation:
    ```
    # 'update_seg_table' must be importable from 'scripts.attributes'
    add_segmentation(source_name, seg_name,
                     paintera_project=('/path/to/paintera/root.n5', '/path/in/file'),
                     resolution=(.025, .02, .02),  # resolution in microns, must be passed for dynamic seg
                     table_update_function='update_seg_table')
    ```

    Paramter:
        source_name [str] - prefix of the primary data source.
        name [str] - name of segmentation data.
        segmentation_path [str] - path to the segmentation that should be added.
            This argument must be specified for static segmentations (= segmentations without paintera project)
            Data needs to be in bdv-hdf format and the path needs to point to the xml. (default: None)
        table_path_dict [dict] - dictionary with table names and paths for this segmentation.
            If given, this must contain an element with name 'default' (default: None).
        paintera_project [tuple[str]] - path and key to paintera project for this segmentation (default: None).
        resolution [listlike[int]] - resolution of this segmentation in microns.
            This only needs to be passed if the segmentation is not static (default: None).
        table_update_function [str] - name of the update function that will be called when
            the segmentation is updated from paintera corrections (default: None).
        copy_data [bool] - whether to copy the data. This should be set to True,
            unless adding a segmentation that is already in the rawdata folder. (default: True)
        is_private [bool] - whether this data is private (default: False).
    """
    # validate the inputs

    # validate the source name and segmentation name
    source_names = get_source_names()
    if source_name not in source_names:
        raise ValueError("""Source %s is not in the current sources.
                            Use 'add_source' to add a new source.""" % source_name)
    output_name = '%s-%s' % (source_name, name)
    names = get_segmentation_names()
    if output_name in names:
        raise ValueError("Name %s is already taken" % output_name)

    # validate the combination of arguments
    if not ((segmentation_path is None) != (paintera_project is None)):
        raise ValueError("Expect either one of segmentation path or paintera project to be set.")
    is_static = segmentation_path is not None
    if not is_static:
        if resolution is None or len(resolution) != 3:
            raise ValueError("Invalid combination: you have passed a non-static segmenation, but no resolution.")
    if table_path_dict and not is_static:
        raise ValueError("Invalid combination: you have passed a table and a non-static segmentation.")
    if table_update_function and is_static:
        raise ValueError("Invalid combination: you have passed a table update function and a static segmentation.")

    # validate the individual arguments
    if segmentation_path and not check_bdv(segmentation_path):
        raise ValueError("Expect input segmentation to be in bdv format.")
    if table_path_dict and not check_tables(table_path_dict):
        raise ValueError("Expect input table to be in valid csv format.")
    if paintera_project and not check_paintera(paintera_project):
        raise ValueError("Input paintera project is not valid")
    if table_update_function and not check_attributes(table_update_function):
        raise ValueError("The table update function %s is not valid" % str(table_update_function))

    # copy the segmentation data if we have a static segmentation
    if is_static:
        h5_path = get_h5_path_from_xml(segmentation_path, return_absolute_path=True)
        name_h5 = '%s.h5' % output_name
        out_xml = os.path.join(RAW_FOLDER, '%s.xml' % output_name)
        out_h5 = os.path.join(RAW_FOLDER, name_h5)
        if copy_data:
            # copy h5 and xml to the rawdata folder, update the xml with new relative path
            copyfile(h5_path, out_h5)
            copy_xml_with_newpath(segmentation_path, out_xml, name_h5)
        else:
            if not os.path.exists(out_xml) or not os.path.exists(out_h5):
                raise RuntimeError("You specified to not copy the data, but %s and %s do not exist" % (out_xml,
                                                                                                       out_h5))

    # copy the table
    if table_path_dict:
        table_folder = os.path.join(RAW_FOLDER, 'tables', output_name)
        os.makedirs(table_folder, exist_ok=True)
        for name, table_path in table_path_dict.items():
            table_out = os.path.join(table_folder, '%s.csv' % name)
            copyfile(table_path, table_out)

    # register the segmentation
    segmentations = get_segmentations()
    if is_static:
        segmentation = {'is_static': True, 'has_tables': table_path_dict is not None}
    else:
        segmentation = {'is_static': False, 'paintera_project': paintera_project,
                        'resolution': resolution, 'table_update_function': table_update_function}

    segmentations[output_name] = segmentation
    with open(SEGMENTATION_FILE, 'w') as f:
        json.dump(segmentations, f)
    # add the name to the private list if is_private == True
    if is_private:
        add_to_privates(output_name)
