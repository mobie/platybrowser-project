import os


def h5_path_to_xml(h5_path):
    """
    Convert a path to a bdv h5 file, to the path to the xml
    """
    xml_path = os.path.splitext(h5_path)[0] + '.xml'
    return xml_path
