import os
import h5py
import numpy as np

from mmpb.default_config import write_default_global_config
from mmpb.attributes.region_attributes import region_attributes
from mmpb.files.xml_utils import get_h5_path_from_xml


def recompute_table(version):
    version_folder = '../../data/%s' % version
    image_folder = os.path.join(version_folder, 'images')
    seg_folder = os.path.join(version_folder, 'segmentations')
    table_folder = os.path.join(version_folder, 'tables')

    seg_path = os.path.join(seg_folder, 'sbem-6dpf-1-whole-segmented-cells-labels.xml')
    seg_path = get_h5_path_from_xml(seg_path, return_absolute_path=True)

    with h5py.File(seg_path, 'r') as f:
        n_labels = int(f['t00000/s00/0/cells'].attrs['maxId'] + 1)
    label_ids = np.arange(n_labels)

    tmp_folder = './tmp_regions'
    write_default_global_config(os.path.join(tmp_folder, 'configs'))
    target = 'local'
    max_jobs = 48

    region_out = os.path.join(table_folder, 'sbem-6dpf-1-whole-segmented-cells-labels', 'regions.csv')

    region_attributes(seg_path, region_out,
                      image_folder, seg_folder,
                      label_ids, tmp_folder, target, max_jobs)


if __name__ == '__main__':
    recompute_table('0.6.5')
