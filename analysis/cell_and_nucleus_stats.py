import os
import numpy as np
import pandas as pd

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'
CELL_NAME = 'sbem-6dpf-1-whole-segmented-cells'
NUCLEUS_NAME = 'sbem-6dpf-1-whole-segmented-nuclei'


def number_of_cells(version):
    table_path = os.path.join(ROOT, version, 'tables', CELL_NAME, 'default.csv')
    table = pd.read_csv(table_path, sep='\t')
    col = table['cells'].values
    n_cells = np.sum(col == 1)
    n_segmented_objects = len(col)
    print(n_cells, "out of", n_segmented_objects,
          "segmented objcets are classified as cells, because they contain a uniquely mapped nucleus")


def _compute_sizes(n_pixels, resolution, percentile):
    pixel_vol_um = np.prod(resolution)

    cell_volumes = n_pixels * pixel_vol_um
    # remove outliers
    if percentile is not None:
        min_size = np.percentile(cell_volumes, percentile)
        max_size = np.percentile(cell_volumes, 100 - percentile)

        cell_volumes = cell_volumes[cell_volumes > min_size]
        cell_volumes = cell_volumes[cell_volumes < max_size]

    return cell_volumes


def cell_sizes(version, percentile=1):
    table_path = os.path.join(ROOT, version, 'tables', CELL_NAME, 'default.csv')
    table = pd.read_csv(table_path, sep='\t')

    n_pixels = table['n_pixels'].values
    cells = table['cells'].values.astype('bool')
    n_pixels = n_pixels[cells]

    resolution = [0.025, 0.02, 0.02]
    cell_volumes_um = _compute_sizes(n_pixels, resolution, percentile)
    print("Min cell size:", cell_volumes_um.min(), "(um)^3")
    print("Max cell size:", cell_volumes_um.max(), "(um)^3")
    print("Mean cell size:", cell_volumes_um.mean(), "+-", cell_volumes_um.std(), "(um)^3")


def nucleus_sizes(version, percentile=1):
    table_path = os.path.join(ROOT, version, 'tables', NUCLEUS_NAME, 'default.csv')
    table = pd.read_csv(table_path, sep='\t')

    n_pixels = table['n_pixels'].values[1:]

    resolution = [0.1, 0.08, 0.08]
    nucleus_volumes_um = _compute_sizes(n_pixels, resolution, percentile)
    print("Min nucleus size:", nucleus_volumes_um.min(), "(um)^3")
    print("Max nucleus size:", nucleus_volumes_um.max(), "(um)^3")
    print("Mean nucleus size:", nucleus_volumes_um.mean(), "+-", nucleus_volumes_um.std(), "(um)^3")


def cells_per_region(root_folder):
    table_path = os.path.join(ROOT, version, 'tables', CELL_NAME, 'default.csv')
    table = pd.read_csv(table_path, sep='\t')
    cells = table['cells'].values.astype('bool')
    n_cells = float(cells.sum())

    region_table_path = os.path.join(ROOT, version, 'tables', CELL_NAME, 'regions.csv')
    region_table = pd.read_csv(region_table_path, sep='\t')
    assert len(region_table) == len(table)

    # names of the regions segmented in propspr + muscle and gland segmentation
    region_names = ['allglands',
                    'crypticsegment',
                    'foregut',
                    'gut',
                    'head',
                    'lateralectoderm',
                    'midgut',
                    'muscle',
                    'pygidium',
                    'vnc']
    for name in region_names:
        in_region = region_table[name].values[cells]
        assert in_region.min() == 0 and in_region.max() == 1
        n_region = float(in_region.sum())
        print("Number of cells", name, ":")
        print(n_region, "=", round(n_region / n_cells * 100, 1), "%")


if __name__ == '__main__':
    version = '0.6.6'
    number_of_cells(version)
    print()
    cell_sizes(version)
    print()
    nucleus_sizes(version)
    print()
    cells_per_region(version)
