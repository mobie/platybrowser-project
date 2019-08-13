import json

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


def add_image(input_path):
    """ Add image volume to the platy browser data.
    """
    pass


def add_segmentation():
    """ Add segmentation volume to the platy browser data.
    """
    pass
