import os
import json
from glob import glob
from mmpb.files.copy_helper import copy_attributes
from pybdv.util import get_key, get_number_of_scales

ROOT = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


def fix_dynamic_seg_dict(version_folder):
    dict_path = os.path.join(version_folder, 'misc', 'dynamic_segmentations.json')
    with open(dict_path) as f:
        seg_dict = json.load(f)

    for name, props in seg_dict.items():
        if 'PainteraProject:' in props:
            pproject = props.pop('PainteraProject:')
            props['PainteraProject'] = pproject
        if 'cilia' in name:
            props['MapToBackground'] = [1]
        seg_dict[name] = props

    with open(dict_path, 'w') as f:
        json.dump(seg_dict, f, indent=2, sort_keys=True)


def fix_all_dynamic_seg_dicts():
    vfolders = glob(os.path.join(ROOT, '0.*'))
    for folder in vfolders:
        fix_dynamic_seg_dict(folder)


def fix_copy_attributes():
    copied_path = '/g/kreshuk/pape/copied_to_n5.json'
    with open(copied_path) as f:
        copied_files = json.load(f)

    for h5_path in copied_files:
        n5_path = os.path.splitext(h5_path)[0] + '.n5'
        n_scales = get_number_of_scales(n5_path, 0, 0)
        for scale in range(n_scales):
            copy_attributes(h5_path, get_key(True, 0, 0, scale),
                            n5_path, get_key(False, 0, 0, scale))


if __name__ == '__main__':
    # fix_all_dynamic_seg_dicts()
    fix_copy_attributes()
