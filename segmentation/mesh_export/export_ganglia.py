import argparse
import os

import pandas as pd
from mmpb.export.meshes import export_meshes

ROOT = '../../data'


def get_ganglia_ids(version):
    # get the normal table to check which ones are actually cells
    table = os.path.join(ROOT, version, 'tables',
                         'sbem-6dpf-1-whole-segmented-ganglia', 'default.csv')
    table = pd.read_csv(table, sep='\t')
    return table['label_id'].values.astype('uint32')[1:]


def ganglia_meshes(version, out_folder, scale=1, n_jobs=16):
    ids = get_ganglia_ids(version)

    # load the segmentation dataset
    xml_path = os.path.join(ROOT, version, 'images/local/sbem-6dpf-1-whole-segmented-ganglia.xml')
    table_path = os.path.join(ROOT, version, 'tables/sbem-6dpf-1-whole-segmented-ganglia/default.csv')

    export_meshes(xml_path, table_path, ids, out_folder, scale, n_jobs=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export meshes for ganglia")
    parser.add_argument('--version', type=str, default='1.0.1')
    args = parser.parse_args()

    version = args.version
    out_folder = './meshes_ganglia'
    ganglia_meshes(version, out_folder)
