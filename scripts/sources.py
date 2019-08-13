import json
from.files import check_bdv

SOURCE_FILE = '../data/sources.json'
SEGMENTATION_FILE = '../data/segmentations.json'
IMAGE_FILE = '../data/images.json'


def get_sources():
    """ Get names of the current data sources.

    See https://git.embl.de/tischer/platy-browser-tables/README.md#file-naming
    for the source naming conventions.
    """
    with open(SOURCE_FILE) as f:
        sources = json.load(f)
    return sources


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
    sources.append({'modality': modality, 'stage': stage, 'id': str(id), 'region': region})
    with open(SOURCE_FILE, 'w') as f:
        json.dump(sources, f)


def source_to_prefix(source):
    return '%s-%s-%s-%s' % (source['modality'],
                            source['stage'],
                            source['id'],
                            source['region'])


def get_name_prefixes():
    """ Get the name prefixes corresponding to all sources.
    """
    sources = get_sources()
    prefixes = [source_to_prefix(source) for source in sources]
    return prefixes


def load_image_names():
    with open(IMAGE_FILE) as f:
        names = json.load(f)
    return names


def add_image(input_path, source_prefix, name):
    """ Add image volume to the platy browser data.

    Parameter:
        input_path [str] - path to the data that should be added.
            Needs to be in bdv data format.
        source_prefix [str] - prefix of the primary data source.
        name [str] - name of the data.
    """
    # validate the inputs
    prefixes = get_name_prefixes()
    if source_prefix not in prefixes:
        raise ValueError("""Source prefix %s is not in the current sources.
                            Use 'add_source' to add a new source.""" % source_prefix)
    is_bdv = check_bdv(input_path)
    if not is_bdv:
        raise ValueError("Expect input to be in bdv format")
    output_name = '%s-%s' % (source_prefix, name)
    names = load_image_names()
    if output_name in names:
        raise ValueError("Name %s is already taken" % output_name)

    # TODO copy h5 and xml to the rawdata folder, update the xml with new relative path

    # add name to the name list and serialze
    names.append(output_name)
    with open(IMAGE_FILE, 'w') as f:
        json.dump(names, f)


def load_segmentation_names():
    with open(SEGMENTATION_FILE) as f:
        names = list(json.load(f).keys())
    return names


def add_segmentation(source_prefix, name):
    """ Add segmentation volume to the platy browser data.
    """
    pass
