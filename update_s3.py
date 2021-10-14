import os
import mobie
import xml.etree.ElementTree as ET
from pybdv.metadata import indent_xml


def update_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img = root.find("SequenceDescription").find("ImageLoader")

    endpoint = img.find("ServiceEndpoint")
    endpoint.text = "https://s4.embl.de"

    key = img.find("Key")
    key.text = key.text.replace("remote", "local")

    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_file)


def update_s3():
    ds_folder = "./data/1.0.1"
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]
    for name, source in sources.items():
        source = source[list(source.keys())[0]]
        xml_file = os.path.join(ds_folder, source["imageData"]["bdv.n5.s3"]["relativePath"])
        assert os.path.exists(xml_file), xml_file
        update_xml(xml_file)


if __name__ == "__main__":
    update_s3()
