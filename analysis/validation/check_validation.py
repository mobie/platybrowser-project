import h5py
from heimdall import view, to_source
from scripts.segmentation.validation import eval_cells, eval_nuclei, get_ignore_seg_ids


def check_cell_evaluation():
    from scripts.segmentation.validation.eval_cells import (eval_slice,
                                                            get_bounding_box)

    praw = '../../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    pseg = '../../data/0.5.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    pann = '../../data/rawdata/evaluation/validation_annotations.h5'

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
    pseg = '../../data/0.5.5/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
    pann = '../../data/rawdata/evaluation/validation_annotations.h5'

    table_path = '../../data/0.5.5/tables/sbem-6dpf-1-whole-segmented-cells-labels/regions.csv'
    ignore_seg_ids = get_ignore_seg_ids(table_path)

    res = eval_cells(pseg, 't00000/s00/0/cells', pann,
                     ignore_seg_ids=ignore_seg_ids)
    print("Eval result:")
    print(res)


# TODO
def check_nucleus_evaluation():
    pass


# TODO
def eval_all_nulcei():
    eval_nuclei()


if __name__ == '__main__':
    # check_cell_evaluation()
    eval_all_cells()
