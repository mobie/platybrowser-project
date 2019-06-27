import os
import glob
from scripts.files import get_h5_path_from_xml, copy_xml_with_newpath, write_simple_xml
from shutil import copyfile


def copy_xmls_and_symlink_h5(name_dict, src_folder, trgt_folder):
    for n1, n2 in name_dict.items():
        src_xml = os.path.join(src_folder, n1)
        trgt_xml = os.path.join(trgt_folder, n2)

        # we make a softlink from src to target h5
        # NOTE eventually, we want to copy all the data, but
        # for now, we use softlinks in order to keep the current
        # version of the platy browser working
        src_h5 = get_h5_path_from_xml(src_xml, return_absolute_path=True)
        trgt_h5 = os.path.splitext(n2)[0] + '.h5'
        os.symlink(src_h5, os.path.join(trgt_folder, trgt_h5))

        copy_xml_with_newpath(src_xml, trgt_xml, trgt_h5)


def make_initial_data_sources(copy_sbem,
                              copy_prospr,
                              copy_fib,
                              copy_regions):
    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    raw_folder = './data/rawdata'
    os.makedirs(raw_folder, exist_ok=True)

    # TODO
    # copy cellular models
    if copy_sbem:
        print("Copy sbem data")
        # copy the sbem data
        sbem_prefix = 'sbem-6dpf-1-whole'
        name_dict = {'em-raw-full-res.xml': 'raw.xml',
                     'em-segmented-muscles-ariadne.xml': 'segmented-muscle.xml',
                     'em-segmented-tissue-labels.xml': 'segmented-tissue-labels.xml'}
        name_dict = {k: '%s-%s' % (sbem_prefix, v)
                     for k, v in name_dict.items()}
        copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)

    if copy_prospr:
        print("Copy prospr data")
        prospr_prefix = 'prospr-6dpf-1-whole'
        # copy the prospr meds
        prospr_names = glob.glob(os.path.join(old_folder, "*-MED*"))
        prospr_names = [os.path.split(f)[1] for f in prospr_names]
        prospr_names = [name for name in prospr_names if os.path.splitext(name)[1] == '.xml']
        name_dict = {n: '%s-%s' % (prospr_prefix, n) for n in prospr_names}
        copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)

        # copy valentyna's med file
        input_path = '/g/kreshuk/zinchenk/cell_match/data/meds_all_genes_500nm.h5'
        output_path = os.path.join(raw_folder, '%s_meds_all_genes.h5' % prospr_prefix)
        xml_path = os.path.join(raw_folder, '%s_meds_all_genes.xml' % prospr_prefix)
        write_simple_xml(xml_path, os.path.split(output_path)[1],
                         path_type='relative')
        copyfile(input_path, output_path)

    if copy_fib:
        print("Coby fibsem data")
        # copy the fibsem data
        fib_prefix = 'fibsem-6dpf-1-parapod'
        name_dict = {'em-raw-': 'em-raw',
                     'em-segmented-': ''}
        name_dict = {k: '%s-%s' % (fib_prefix, v)
                     for k, v in name_dict.items()}
        copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)

    if copy_regions:

        print("Copy regions")
        prospr_prefix = 'prospr-6dpf-1-whole-segmented'
        prospr_names = glob.glob(os.path.join(old_folder, "BodyPart_*"))
        prospr_names = [os.path.split(f)[1] for f in prospr_names]
        prospr_names = [name for name in prospr_names if os.path.splitext(name)[1] == '.xml']
        name_dict = {n: '%s-%s' % (prospr_prefix, n.split('_')[-1]) for n in prospr_names}
        copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)


if __name__ == '__main__':
    copy_sbem = False
    copy_prospr = False
    copy_fib = False
    copy_regions = True
    make_initial_data_sources(copy_sbem, copy_prospr, copy_fib,
                              copy_regions)
