import os
# from pathlib import Path
from shutil import copyfile


# TODO need to replace the name in xml to make this work out of the box
def export_segmentation(folder, name, dest_folder, out_name):
    seg_file = os.path.join(folder, 'segmentations', '%s.h5' % name)
    xml_file = os.path.join(folder, 'segmentations', '%s.xml' % name)
    table_file = os.path.join(folder, 'tables', name, 'default.csv')
    assert os.path.exists(seg_file)
    assert os.path.exists(xml_file)
    assert os.path.exists(table_file), table_file

    seg_out = os.path.join(dest_folder, '%s.h5' % out_name)
    print("Copying segmentation from", seg_file, "to", seg_out)
    copyfile(seg_file, seg_out)

    xml_out = os.path.join(dest_folder, '%s.xml' % out_name)
    print("Copying xml from", xml_file, "to", xml_out)
    copyfile(xml_file, xml_out)

    table_out = os.path.join(dest_folder, 'tables', '%s.csv' % out_name)
    print("Copying table from", table_file, "to", table_out)
    copyfile(table_file, table_out)


def export_tag(tag, export_cells=False, export_cilia=False):
    tag_folder = './data/%s' % tag
    assert os.path.exists(tag_folder)

    dest_folder = '/g/arendt/EM_6dpf_segmentation/EM-Prospr'

    if export_cells:
        name = 'sbem-6dpf-1-whole-segmented-cells-labels'
        out_name = 'em-segmented-cells-new-labels'
        export_segmentation(tag_folder, name, dest_folder, out_name)

    if export_cilia:
        name = 'sbem-6dpf-1-whole-segmented-cilia-labels'
        out_name = 'em-segmented-cilia-labels'
        export_segmentation(tag_folder, name, dest_folder, out_name)


if __name__ == '__main__':
    # export_tag('0.1.1', export_cells=True)
    export_tag('0.1.0', export_cilia=True)
