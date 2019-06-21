import os
import glob
from .scripts.files import get_h5_path_from_xml, copy_xml_with_newpath


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
        os.symlink(src_h5, trgt_h5)

        copy_xml_with_newpath(src_xml, trgt_xml, trgt_h5)


def make_initial_data_sources():
    old_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'
    raw_folder = './data/rawdata'

    # copy the sbem data
    sbem_prefix = 'sbem-6dpf-1-whole'
    name_dict = {'em-raw-', 'em-raw',
                 'em-segmented-', ''}
    name_dict = {k: '%s-%s' % (sbem_prefix, v)
                 for k, v in name_dict.items()}
    copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)

    # copy the prospr meds
    prospr_prefix = 'prospr-6dpf-1-whole'
    prospr_names = glob.glob(os.path.join(old_folder, "*-MED*"))
    prospr_names = [os.path.split(f) for f in prospr_names]
    name_dict = {n: '%s-%s' % (prospr_prefix, n) for n in prospr_names}
    copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)

    # copy the fibsem data
    fib_prefix = 'fibsem-6dpf-1-parapod'
    name_dict = {'em-raw-', 'em-raw',
                 'em-segmented-', ''}
    name_dict = {k: '%s-%s' % (fib_prefix, v)
                 for k, v in name_dict.items()}
    copy_xmls_and_symlink_h5(name_dict, old_folder, raw_folder)


if __name__ == '__main__':
    make_initial_data_sources()
