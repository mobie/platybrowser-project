import os
import json
from glob import glob
from mmpb.files.copy_helper import copy_attributes
from mmpb.attributes.genes import create_auxiliary_gene_file
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


def fix_id_luts(version_folder):
    id_luts = glob(os.path.join(version_folder, 'misc', 'new_id_lut*'))
    for id_lut in id_luts:

        # only need to correct if old labels tag is still in the name
        if 'labels' not in id_lut:
            continue

        new_id_lut = id_lut.replace('-labels', '')
        if os.path.islink(id_lut):
            link_loc = os.readlink(id_lut)
            new_link_loc = link_loc.replace('-labels', '')
            os.unlink(id_lut)
            os.symlink(new_link_loc, new_id_lut)
        else:
            os.rename(id_lut, new_id_lut)


def fix_all_id_luts():
    vfolders = glob(os.path.join(ROOT, '0.*'))
    for folder in vfolders:
        fix_id_luts(folder)


def add_remote_storage_to_image_dict():
    version = '0.6.6'
    im_dict_path = os.path.join(ROOT, version, 'images', 'images.json')
    with open(im_dict_path, 'r') as f:
        im_dict = json.load(f)

    for name, props in im_dict.items():
        storage = props['Storage']
        rel_remote = os.path.join('remote', name + '.xml')
        abs_remote = os.path.join(ROOT, version, 'images', rel_remote)
        assert os.path.exists(abs_remote), abs_remote
        storage['remote'] = rel_remote
        props['Storage'] = storage
        im_dict[name] = props

    with open(im_dict_path, 'w') as f:
        json.dump(im_dict, f, indent=2, sort_keys=True)


def rewrite_gene_file():
    meds_root = '../data/0.6.3/images/local'
    out_file = '../data/0.6.3/misc/prospr-6dpf-1-whole_meds_all_genes.h5'
    create_auxiliary_gene_file(meds_root, out_file)


def update_image_dict(version):
    """ https://github.com/platybrowser/platybrowser-backend/issues/10
    """
    image_dict_path = os.path.join(ROOT, version, 'images', 'images.json')
    with open(image_dict_path) as f:
        image_dict = json.load(f)

    for name, props in image_dict.items():
        # for sbem
        if name.startswith('sbem'):
            # raw data modality
            if 'raw' in name:
                props['Type'] = 'Image'
            elif 'cells' in name or\
                    'chromatin' in name or\
                    'cilia' in name or\
                    'ganglia' in name or\
                    'nephridia' in name or\
                    'nuclei' in name or\
                    'tissue' in name:
                props['Type'] = 'Segmentation'
                props['MinValue'] = 0
                props['MaxValue'] = 1000
                props['ColorMap'] = 'Glasbey'
                props.pop('Color', None)
            else:
                props['Type'] = 'Mask'
                props['MinValue'] = 0
                props['MaxValue'] = 1
        # for prospr
        elif name.startswith('prospr'):
            if 'virtual' in name:
                props['Type'] = 'Segmentation'
                props['MinValue'] = 0
                props['MaxValue'] = 1000
            elif 'segmented' in name:
                props['Type'] = 'Mask'
                props['MinValue'] = 0
                props['MaxValue'] = 1000
            else:
                props['Type'] = 'Image'
                props['Color'] = 'RandomFromGlasbey'
        else:
            raise RuntimeError('Unknown name %s' % name)

        image_dict[name] = props

    with open(image_dict_path, 'w') as f:
        json.dump(image_dict, f, sort_keys=True, indent=2)


def update_all_image_dicts():
    version_file = '../data/versions.json'
    with open(version_file) as f:
        versions = json.load(f)

    for version in versions:
        if version == '0.6.6':
            continue
        update_image_dict(version)


if __name__ == '__main__':
    # fix_all_dynamic_seg_dicts()
    # fix_copy_attributes()
    # fix_all_id_luts()
    # add_remote_storage_to_image_dict()
    # rewrite_gene_file()
    update_image_dict('0.6.6')
    # update_all_image_dicts()
