import os
import json
from glob import glob

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


if __name__ == '__main__':
    fix_all_dynamic_seg_dicts()
