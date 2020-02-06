import os
from mmpb.segmentation.validation.refine_annotations import refine, export_refined
from mmpb.segmentation.validation.eval_cells import get_ignore_seg_ids


def proofread(orientation, slice_id):
    seg_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.1',
                            'segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    seg_key = 't00000/s00/0/cells'
    table_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.1',
                              'tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv')
    ignore_seg_ids = get_ignore_seg_ids(table_path)

    proj_folder = 'project_folders/proofread_%s_%i' % (orientation, slice_id)
    refine(seg_path, seg_key, ignore_seg_ids,
           orientation, slice_id, proj_folder)


def export_slices(orientation, slice_ids):
    out_path = '../../data/rawdata/evaluation/validation_annotations.h5'
    for slice_id in slice_ids:
        out_key = '%s/%i' % (orientation, slice_id)
        project_folder = 'project_folders/proofread_%s_%i' % (orientation, slice_id)
        export_refined(project_folder, out_path, out_key)


# orientation xy has slices:
# 1000, 2000, 4000, 7000
# orientation xz has slices:
# 4000, 5998, 8662, 9328
if __name__ == '__main__':

    orientation = 'xy'
    slice_id = 4000
    proofread(orientation, slice_id)

    # slices_xy = [1000, 2000, 4000, 7000]
    # slices_xz = [4000, 5998, 8662, 9328]
    # export_slices('xy', slices_xy)
    # export_slices('xz', slices_xz)
