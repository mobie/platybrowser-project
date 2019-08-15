import os
from .xml_utils import get_h5_path_from_xml


# TODO check more attributes in the xml to make sure that this actually is
# a bdv format file
def check_bdv(path):
    ext = os.path.splitext(path)[1]
    if ext != '.xml':
        return False
    h5_path = get_h5_path_from_xml(path, return_absolute_path=True)
    if not os.path.exists(h5_path):
        return False
    return True
