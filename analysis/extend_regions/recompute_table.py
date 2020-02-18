import os
import h5py
import numpy as np

from pybdv.util import get_key
from mmpb.attributes.region_attributes import region_attributes
from mmpb.default_config import write_default_global_config
from mmpb.files.xml_utils import get_h5_path_from_xml
from mmpb.util import is_h5_file


def recompute_table(version):
    version_folder = '../../data/%s' % version
    image_folder = os.path.join(version_folder, 'images')
    seg_folder = os.path.join(version_folder, 'segmentations')
    table_folder = os.path.join(version_folder, 'tables')

    seg_path = os.path.join(seg_folder, 'sbem-6dpf-1-whole-segmented-cells.xml')
    seg_path = get_h5_path_from_xml(seg_path, return_absolute_path=True)
    is_h5 = is_h5_file(seg_path)
    key = get_key(is_h5, setup_id=0, time_point=0, scale=0)

    with h5py.File(seg_path, 'r') as f:
        n_labels = int(f[key].attrs['maxId'] + 1)
    label_ids = np.arange(n_labels)

    tmp_folder = './tmp_regions'
    write_default_global_config(os.path.join(tmp_folder, 'configs'))
    target = 'local'
    max_jobs = 48

    region_out = os.path.join(table_folder, 'sbem-6dpf-1-whole-segmented-cells', 'regions.csv')

    region_attributes(seg_path, region_out,
                      image_folder, seg_folder,
                      label_ids, tmp_folder, target, max_jobs)


if __name__ == '__main__':
    recompute_table('0.6.5')
