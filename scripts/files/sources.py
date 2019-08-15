import json
import os
from shutil import copyfile
from .checks import check_bdv
from .xml_utils import get_h5_path_from_xml, copy_xml_with_newpath

RAW_FOLDER = 'data/rawdata'
SOURCE_FILE = 'data/sources.json'
SEGMENTATION_FILE = 'data/segmentations.json'
IMAGE_FILE = 'data/images.json'


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


def get_segmentation_names():
    if not os.path.exists(SEGMENTATION_FILE):
        return []
    with open(SEGMENTATION_FILE) as f:
        names = list(json.load(f).keys())
    return names


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


def add_image(input_path, source_name, name, copy_data=True):
    """ Add image volume to the platy browser data.

    Parameter:
        input_path [str] - path to the data that should be added.
            Data needs to be in bdv-hdf5 format and the path needs to point to the xml.
        source_prefix [str] - prefix of the primary data source.
        name [str] - name of the data.
        copy_data [bool] - whether to copy the data. This should be set to True,
            unless adding an image that is already in the rawdata folder. (default: True)
    """
    # validate the inputs
    source_names = get_source_names()
    if source_name not in source_names:
        raise ValueError("""Source %s is not in the current sources.
                            Use 'add_source' to add a new source.""" % source_name)
    is_bdv = check_bdv(input_path)
    if not is_bdv:
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


def add_segmentation(source_prefix, name):
    """ Add segmentation volume to the platy browser data.
    """
    pass
