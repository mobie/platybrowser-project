import xml.etree.ElementTree as ET


def get_h5_path_from_xml(xml_path):
    # xml horror ...
    et_root = ET.parse(xml_path).getroot()
    et = et_root[1]
    et = et[0]
    et = et[0]
    path = et.text
    return path
