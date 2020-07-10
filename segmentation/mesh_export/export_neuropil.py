import argparse
import os

import pandas as pd
from mmpb.export.meshes import export_meshes

ROOT = '../../data'


def neuropil_mesh(version, out_folder, scale=2):
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-neuropil.xml')
    table_path = None
    ids = [255]
    export_meshes(xml_path, table_path, ids, out_folder, scale)


def neuropil_mesh_head(version, out_folder, scale=3):
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-neuropil.xml')
    xml_mask_path = os.path.join(ROOT, version, 'images/local/prospr-6dpf-1-whole-segmented-head.xml')
    table_path = None
    ids = [255]
    export_meshes(xml_path, table_path, ids, out_folder, scale,
                  xml_mask_path=xml_mask_path, mask_scale=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export meshes for neuropil")
    parser.add_argument('--version', type=str, default='1.0.1')
    parser.add_argument('--intersect_head', type=str, default=0)
    args = parser.parse_args()

    version = args.version
    if bool(args.intersect_head):
        out_folder = './meshes_neuropil_head'
        neuropil_mesh_head(version, out_folder)
    else:
        out_folder = './meshes_neuropil'
        neuropil_mesh(version, out_folder)
