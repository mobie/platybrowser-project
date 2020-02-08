import os
import json
import h5py
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


def replace_gene_names_h5():
    path = '../data/0.6.3/misc/prospr-6dpf-1-whole_meds_all_genes.h5'

    ref_gene_names = glob('../data/0.6.3/images/local/prospr*.xml')
    ref_gene_names = [os.path.splitext(os.path.split(name)[1])[0]
                      for name in ref_gene_names]
    ref_gene_names = ['-'.join(name.split('-')[4:]) for name in ref_gene_names]
    ref_gene_names = [name for name in ref_gene_names if 'segmented' not in name]
    ref_gene_names = [name for name in ref_gene_names if 'virtual' not in name]

    def replace_gene_name(name):
        new_name = name.split('-')
        if new_name[0].startswith('ENR') or new_name[0].startswith('NOV'):
            new_name = new_name[1:]
        new_name = '-'.join(new_name).lower()
        return new_name

    with h5py.File(path) as f:
        ds = f['gene_names']
        gene_names = [i.decode('utf-8') for i in ds]
        gene_names = [replace_gene_name(name) for name in gene_names]

        assert len(gene_names) == len(ref_gene_names)
        print(set(gene_names) - set(ref_gene_names))
        print(set(ref_gene_names) - set(gene_names))
        assert len(set(gene_names) - set(ref_gene_names)) == 0

        # TODO once all gene names agree
        # gene_names_ascii = [n.encode('ascii', 'ignore') for n in gene_names]
        # f.create_dataset(names_dset, data=gene_names_ascii, dtype='S40')


if __name__ == '__main__':
    # fix_all_dynamic_seg_dicts()
    # fix_copy_attributes()
    # fix_all_id_luts()
    replace_gene_names_h5()
