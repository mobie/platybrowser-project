import argparse
import os
from mmpb.export.meshes import export_meshes

ROOT = '../../data'


def surface_mesh(version, out_folder, scale=0):
    # load the segmentation dataset
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-inside.xml')
    table_path = None

    cell_ids = [255]
    export_meshes(xml_path, table_path, cell_ids, out_folder, scale, resolution=None, n_jobs=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export meshes for the surface")
    parser.add_argument('--version', type=str, default='1.0.1')
    args = parser.parse_args()

    version = args.version
    out_folder = './meshes_surface'
    surface_mesh(version, out_folder)
