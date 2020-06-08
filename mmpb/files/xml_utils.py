import xml.etree.ElementTree as ET
from pybdv.metadata import indent_xml


def write_simple_xml(xml_path, data_path, path_type='absolute'):
    # write top-level data
    root = ET.Element('SpimData')
    root.set('version', '0.2')
    bp = ET.SubElement(root, 'BasePath')
    bp.set('type', 'relative')
    bp.text = '.'

    seqdesc = ET.SubElement(root, 'SequenceDescription')
    imgload = ET.SubElement(seqdesc, 'ImageLoader')
    imgload.set('format', 'bdv.hdf5')
    el = ET.SubElement(imgload, 'hdf5')
    el.set('type', path_type)
    el.text = data_path

    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)
