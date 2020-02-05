import os
import json
import vigra
import numpy as np
import napari
from heimdall import view, to_source
from elf.io import open_file

from .eval_cells import get_bounding_box
from .evaluate_annotations import evaluate_annotations

DEFAULT_ANNOTATION_PATH = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                                       'rawdata/evaluation/validation_annotations.h5')
DEFAULT_RAW_PATH = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.h5'


def compute_masks(seg, labels, ignore_seg_ids):

    seg_eval = vigra.analysis.labelImageWithBackground(seg)

    if ignore_seg_ids is None:
        this_ignore_ids = None
    else:
        ignore_mask = np.isin(seg, ignore_seg_ids)
        this_ignore_ids = np.unique(seg_eval[ignore_mask])

    fg_annotations = np.isin(labels, [1, 2]).astype('uint32')
    bg_annotations = labels == 3

    min_radius = 16
    _, masks = evaluate_annotations(seg_eval, fg_annotations, bg_annotations,
                                    this_ignore_ids, min_radius=min_radius,
                                    return_masks=True)
    return masks['merges'], masks['splits']


def refine(seg_path, seg_key, ignore_seg_ids,
           orientation, slice_id,
           project_folder,
           annotation_path=DEFAULT_ANNOTATION_PATH,
           raw_path=DEFAULT_RAW_PATH,
           raw_key='t00000/s00/1/cells'):

    label_path = os.path.join(project_folder, 'labels.npy')
    fm_path = os.path.join(project_folder, 'fm.npy')
    fs_path = os.path.join(project_folder, 'fs.npy')
    bb_path = os.path.join(project_folder, 'bounding_box.json')

    if os.path.exists(project_folder):
        print("Load from existing project")
        labels = np.load(label_path) if os.path.exists(label_path) else None
        fm = np.load(fm_path) if os.path.exists(fm_path) else None
        fs = np.load(fs_path) if os.path.exists(fs_path) else None
    else:
        print("Start new project")
        labels, fm, fs = None, None, None

    with open_file(annotation_path, 'r') as fval:
        ds = fval[orientation][str(slice_id)]
        bb = get_bounding_box(ds)
        ds.n_threads = 8
        if labels is None:
            labels = ds[:]

    starts = [int(b.start) for b in bb]
    stops = [int(b.stop) for b in bb]

    with open_file(seg_path, 'r') as f:
        ds = f[seg_key]
        ds.n_threads = 8
        seg = ds[bb].squeeze().astype('uint32')

    with open_file(raw_path, 'r') as f:
        ds = f[raw_key]
        ds.n_threads = 8
        raw = ds[bb].squeeze()

    assert labels.shape == seg.shape == raw.shape
    if fm is None:
        assert fs is None
        fm, fs = compute_masks(seg, labels, ignore_seg_ids)
    else:
        assert fs is not None

    with napari.gui_qt():
        viewer = view(to_source(raw, name='raw'), to_source(labels, name='labels'),
                      to_source(seg, name='seg'), to_source(fm, name='merges'),
                      to_source(fs, name='splits'), return_viewer=True)

        @viewer.bind_key('s')
        def save_labels(viewer):
            print("Saving state ...")
            layers = viewer.layers
            os.makedirs(project_folder, exist_ok=True)

            labels = layers['labels'].data
            np.save(label_path, labels)

            fm = layers['merges'].data
            np.save(fm_path, fm)

            fs = layers['splits'].data
            np.save(fs_path, fs)

            with open(bb_path, 'w') as f:
                json.dump({'starts': starts, 'stops': stops}, f)
            print("... done")


def export_refined(project_folder, out_path, out_key):

    print("Export", project_folder, "to", out_path, out_key)
    label_path = os.path.join(project_folder, 'labels.npy')
    labels = np.load(label_path)

    bb_path = os.path.join(project_folder, 'bounding_box.json')
    with open(bb_path) as f:
        bb = json.load(f)

    with open_file(out_path) as out:
        dso = out.create_dataset(out_key, data=labels, compression='gzip')
        dso.attrs['starts'] = bb['starts']
        dso.attrs['stops'] = bb['stops']
