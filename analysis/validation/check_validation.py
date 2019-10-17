import h5py
from heimdall import view, to_source


def check_cell_evaluation():
    from scripts.segmentation.validation.eval_cells import (eval_slice,
                                                            get_ignore_seg_ids,
                                                            get_bounding_box)

    praw = '../../data'
    pseg = '../../data'
    pann = '../../data'

    table_path = '../../data'
    ignore_seg_ids = get_ignore_seg_ids(table_path)

    with h5py.File(pseg, 'r') as fseg, h5py.File(pann, 'r') as fann:
        ds_seg = fseg['t00000/s00/0/cells']
        ds_ann = fann['xy/']

        print("Run evaluation ...")
        res, masks = eval_slice(ds_seg, ds_ann, ignore_seg_ids, min_radius=16,
                                return_maksks=True)
        fm, fs = masks['false_merges'], masks['false_splits']
        print()
        print("Eval result")
        print(res)
        print()

        print("Load raw data ...")
        bb = get_bounding_box(ds_ann)
        with h5py.File(praw, 'r') as f:
            raw = f['t00000/s00/1/cells'][bb]

        print("Load seg data ...")
        seg = ds_seg[bb].squeeze()

        view(to_source(raw, name='raw'), to_source(seg, name='seg'),
             to_source(fm, name='merges'), to_source(fs, name='splits'))


# def check_nucleus_evaluation():
#     eval_nuclei()


if __name__ == '__main__':
    check_cell_evaluation()
