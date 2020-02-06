import os
import h5py
from heimdall import view, to_source
from mmpb.segmentation.validation import eval_cells, eval_nuclei, get_ignore_seg_ids

# ROOT_FOLDER = '../../data'
ROOT_FOLDER = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data'


def check_cell_evaluation():
    from mmpb.segmentation.validation.eval_cells import (eval_slice,
                                                         get_bounding_box)

    praw = os.path.join(ROOT_FOLDER, 'rawdata/sbem-6dpf-1-whole-raw.h5')
    pseg = os.path.join(ROOT_FOLDER, '0.5.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    pann = os.path.join(ROOT_FOLDER, 'rawdata/evaluation/validation_annotations.h5')

    table_path = '../../data/0.5.5/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv'
    ignore_seg_ids = get_ignore_seg_ids(table_path)

    with h5py.File(pseg, 'r') as fseg, h5py.File(pann, 'r') as fann:
        ds_seg = fseg['t00000/s00/0/cells']
        ds_ann = fann['xy/1000']

        print("Run evaluation ...")
        res, masks = eval_slice(ds_seg, ds_ann, ignore_seg_ids, min_radius=16,
                                return_masks=True)
        fm, fs = masks['merges'], masks['splits']
        print()
        print("Eval result")
        print(res)
        print()

        print("Load raw data ...")
        bb = get_bounding_box(ds_ann)
        with h5py.File(praw, 'r') as f:
            raw = f['t00000/s00/1/cells'][bb].squeeze()

        print("Load seg data ...")
        seg = ds_seg[bb].squeeze().astype('uint32')

        view(to_source(raw, name='raw'), to_source(seg, name='seg'),
             to_source(fm, name='merges'), to_source(fs, name='splits'))


def eval_all_cells():
    pseg = os.path.join(ROOT_FOLDER, '0.5.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    pann = os.path.join(ROOT_FOLDER, 'rawdata/evaluation/validation_annotations.h5')

    table_path = os.path.join(ROOT_FOLDER, '0.5.5/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv')
    ignore_seg_ids = get_ignore_seg_ids(table_path)

    res = eval_cells(pseg, 't00000/s00/0/cells', pann,
                     ignore_seg_ids=ignore_seg_ids)
    print("Eval result:")
    print(res)


def check_nucleus_evaluation():
    from mmpb.segmentation.validation.eval_nuclei import (eval_slice,
                                                          get_bounding_box)

    praw = os.path.join(ROOT_FOLDER, 'rawdata/sbem-6dpf-1-whole-raw.h5')
    pseg = os.path.join(ROOT_FOLDER, '0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5')
    pann = os.path.join(ROOT_FOLDER, 'rawdata/evaluation/validation_annotations.h5')

    with h5py.File(pseg, 'r') as fseg, h5py.File(pann, 'r') as fann:
        ds_seg = fseg['t00000/s00/0/cells']
        ds_ann = fann['xy/1000']

        print("Run evaluation ...")
        res, masks = eval_slice(ds_seg, ds_ann, min_radius=6,
                                return_masks=True)
        fm, fs = masks['merges'], masks['splits']
        print()
        print("Eval result")
        print(res)
        print()

        print("Load raw data ...")
        bb = get_bounding_box(ds_ann, scale_factor=4.)
        with h5py.File(praw, 'r') as f:
            raw = f['t00000/s00/3/cells'][bb].squeeze()

        print("Load seg data ...")
        seg = ds_seg[bb].squeeze().astype('uint32')

        view(to_source(raw, name='raw'), to_source(seg, name='seg'),
             to_source(fm, name='merges'), to_source(fs, name='splits'))


def eval_all_nulcei():
    pseg = os.path.join(ROOT_FOLDER, '0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5')
    pann = os.path.join(ROOT_FOLDER, 'rawdata/evaluation/validation_annotations.h5')

    res = eval_nuclei(pseg, 't00000/s00/0/cells', pann)
    print("Eval result:")
    print(res)


if __name__ == '__main__':
    # check_cell_evaluation()
    eval_all_cells()

    # check_nucleus_evaluation()
    # eval_all_nulcei()
