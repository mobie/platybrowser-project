import os
import xml.etree.ElementTree as ET


# pretty print xml, from:
# http://effbot.org/zone/element-lib.htm#prettyprint
def indent_xml(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def get_h5_path_from_xml(xml_path, return_absolute_path=False):
    # xml horror ...
    et_root = ET.parse(xml_path).getroot()
    et = et_root[1]
    et = et[0]
    et = et[0]
    path = et.text
    # this assumes relative path in xml
    if return_absolute_path:
        path = os.path.join(os.path.split(xml_path)[0], path)
        path = os.path.abspath(os.path.relpath(path))
    return path


def copy_xml_with_abspath(xml_in, xml_out):
    # get the h5 path from the xml
    et_root = ET.parse(xml_in).getroot()
    et = et_root[1]
    et = et[0]
    et = et[0]
    path = et.text

    # NOTE we assume that this is a relative path to the xml's dir
    # would be better to actually read this from the data
    xml_dir = os.path.split(xml_in)[0]
    path = os.path.join(xml_dir, path)
    path = os.path.abspath(os.path.relpath(path))
    if not os.path.exists(path):
        raise RuntimeError("Could not parse proper path from xml")

    # write new xml with the absolute path
    et.text = path
    et.set('type', 'absolute')
    indent_xml(et_root)
    tree = ET.ElementTree(et_root)
    tree.write(xml_out)


def copy_xml_with_newpath(xml_in, xml_out, h5path):
    # get the h5 path from the xml
    et_root = ET.parse(xml_in).getroot()
    et = et_root[1]
    et = et[0]
    et = et[0]
    # write new xml with the new path
    et.text = h5path
    et.set('type', 'absolute')
    indent_xml(et_root)
    tree = ET.ElementTree(et_root)
    tree.write(xml_out)


def write_simple_xml(xml_path, h5_path, path_type='absolute'):
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
    el.text = h5_path

    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)
