#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python

import argparse
import os
import numpy as np
from scripts import get_latest_version
from scripts.analysis import get_region_ids, get_morphology_attribute


def cell_volumes(region_name, version):
    # path are hard-coded, so we need to change the pwd to '..'
    os.chdir('..')
    try:
        if version == '':
            version = get_latest_version()
        region_table_path = "data/%s/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv" % version
        nucleus_table_path = "data/%s/tables/sbem-6dpf-1-whole-segmented-cells-labels/cells_to_nuclei.csv" % version
        label_ids = get_region_ids(region_table_path, nucleus_table_path, region_name)
        morpho_table_path = "data/%s/tables/sbem-6dpf-1-whole-segmented-cells-labels/morphology.csv" % version
        volumes = get_morphology_attribute(morpho_table_path, "shape_volume_in_microns", query_ids=label_ids)
        print("Found average volume for region", region_name, ":")
        print(np.mean(volumes), "+-", np.std(volumes), "cubic micron")
    except Exception as e:
        os.chdir('analysis')
        raise e
    os.chdir('analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average volume of cells for a given region.')
    parser.add_argument('region_name', type=str,
                        help='Name of the region.')
    parser.add_argument('--version', type=str, default='',
                        help='Version of the platy browser data. Default is latest.')

    args = parser.parse_args()
    cell_volumes(args.region_name, args.version)
