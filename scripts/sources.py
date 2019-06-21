# this folder contains information about the current data sources, see
# https://git.embl.de/tischer/platy-browser-tables/blob/dev/README.md#file-naming

# TODO maybe store this as exteral file in json
# list of the current data sources
SOURCES = [{'modality': 'sbem', 'stage': '6dpf', 'id': '1', 'region': 'whole'},
           {'modality': 'prospr', 'stage': '6dpf', 'id': '1', 'region': 'whole'},
           {'modality': 'fibsem', 'stage': '6dpf', 'id': '1', 'region': 'parapod'}]


def get_sources():
    return SOURCES


def source_to_prefix(source):
    return '%s-%s-%s-%s' % (source['modality'],
                            source['stage'],
                            source['id'],
                            source['region'])


def get_name_prefixes():
    sources = get_sources()
    prefixes = [source_to_prefix(source) for source in sources]
    return prefixes
