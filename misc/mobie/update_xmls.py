# TODO integrate this with the xml metadata writer and move it to pybdv
import os
from glob import glob
import xml.etree.ElementTree as ET
from pybdv.metadata import indent_xml


def add_authentication_field(xml_path):
    root = ET.parse(xml_path).getroot()
    loader = root.find("SequenceDescription").find("ImageLoader")
    ET.SubElement(loader, "Authentication").text = "Anonymous"

    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


def update_all_xmls(root):
    folders = glob(os.path.join(root, "*.*"))
    for folder in folders:
        xml_folder = os.path.join(folder, 'images', 'remote')
        xmls = glob(os.path.join(xml_folder, '*.xml'))
        for xml in xmls:
            print(xml)
            add_authentication_field(xml)


if __name__ == '__main__':
    root = '../data'
    update_all_xmls(root)
