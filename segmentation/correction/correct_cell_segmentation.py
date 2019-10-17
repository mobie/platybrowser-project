import os
import json
from scripts.segmentation.correction import (preprocess,
                                             AnnotationTool,
                                             CorrectionTool,
                                             export_node_labels,
                                             to_paintera_format,
                                             rank_false_merges,
                                             get_ignore_ids)


def run_preprocessing(project_folder):
    os.makedirs(project_folder, exist_ok=True)

    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    key = 'volumes/paintera/proofread_cells'
    aff_key = 'volumes/curated_affinities/s1'

    out_path = os.path.join(project_folder, 'data.n5')
    out_key = 'segmentation'

    tissue_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                               'rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.h5')
    tissue_key = 't00000/s00/0/cells'

    preprocess(path, key, aff_key, tissue_path, tissue_key,
               out_path, out_key)


def run_heuristics(project_folder):
    path = os.path.join(project_folder, 'data.n5')
    ignore_ids = get_ignore_ids(path, 'tissue_labels')

    out_path_ids = os.path.join(project_folder, 'fm_candidate_ids.json')
    out_path_scores = os.path.join(project_folder, 'fm_candidate_scores.json')
    n_threads = 32
    n_candidates = 10000

    rank_false_merges(path, 's0/graph', 'features',
                      'morphology', path, 'node_labels',
                      ignore_ids, out_path_ids, out_path_scores,
                      n_threads, n_candidates)


def run_annotations(id_path, project_folder, scale=2,
                    with_node_labels=True):
    p1 = os.path.join(project_folder, 'data.n5')
    table_key = 'morphology'
    scale_factor = 2 ** scale

    p2 = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    rk = 'volumes/raw/s%i' % (scale + 1,)

    if with_node_labels:
        node_label_path = p1
        node_label_key = 'node_labels'
        wsp = p2
        wsk = 'volumes/paintera/proofread_cells/data/s%i' % scale
    else:
        node_label_path, node_label_key = None, None
        wsp = p1
        wsk = 'segmentation/s%i' % scale

    annotator = AnnotationTool(project_folder, id_path,
                               p1, table_key, scale_factor,
                               p2, rk, wsp, wsk,
                               node_label_path, node_label_key)
    annotator()


def filter_fm_ids_by_annotations(project_folder, annotation_path):
    annotation = 'revisit'
    with open(annotation_path) as f:
        annotations = json.load(f)
    annotations = {int(k): v for k, v in annotations.items()}
    ids = [k for k, v in annotations.items() if v == annotation]
    print("Found", len(ids), "ids with annotation", annotation)
    out_path = os.path.join(project_folder, 'fm_ids_filtered.json')
    with open(out_path, 'w') as f:
        json.dump(ids, f)
    return out_path


def run_correction(project_folder, fm_id_path, scale=2):
    p1 = os.path.join(project_folder, 'data.n5')
    table_key = 'morphology'
    segk = 'segmentation/s%i' % scale
    scale_factor = 2 ** scale

    gk = 's0/graph'
    fk = 'features'

    p2 = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    rk = 'volumes/raw/s%i' % (scale + 1,)
    wsk = 'volumes/paintera/proofread_cells/data/s%i' % scale

    # table path
    correcter = CorrectionTool(project_folder, fm_id_path,
                               p1, table_key, scale_factor,
                               p2, rk, p1, segk, p2, wsk,
                               p1, 'node_labels',
                               p1, gk, fk)
    correcter()


def export_correction(project_folder, correct_merges=True, zero_out=True):
    from scripts.segmentation.correction.export_node_labels import zero_out_ids

    p = os.path.join(project_folder, 'data.n5')
    in_key = 'node_labels'
    out_key = 'node_labels_corrected'

    if correct_merges:
        print("Correcting merged ids")
        project_folder = './proj_correct2'
        export_node_labels(p, in_key, out_key, project_folder)
        next_in_key = out_key
    else:
        next_in_key = in_key

    def get_zero_ids(annotation_path):
        annotation = 'merge'
        with open(annotation_path) as f:
            annotations = json.load(f)
        annotations = {int(k): v for k, v in annotations.items()}
        zero_ids = [k for k, v in annotations.items() if v == annotation]
        return zero_ids

    if zero_out:
        # read merge annotations from the morphology annotator
        annotation_path = './proj_annotate_morphology/annotations.json'
        zero_ids = get_zero_ids(annotation_path)
        # read additional merge annotations
        annotation_path = './proj_correct2/annotations.json'
        zero_ids += get_zero_ids(annotation_path)
        print("Zeroing out", len(zero_ids), "ids")

        zero_out_ids(p, next_in_key, p, out_key, zero_ids)

    paintera_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    paintera_key = 'volumes/paintera/proofread_cells_multiset/fragment-segment-assignment'
    print("Exporting to paintera format")
    to_paintera_format(p, out_key, paintera_path, paintera_key)


def correction_workflow_from_ranked_false_merges():
    # the project folder where we store intermediate results
    # for the false merge correction
    project_folder = './project_correct_false_merges'

    # compute the segmentation, the region adjacency graph and the graph weights
    run_preprocessing(project_folder)

    # compute the false merge heuristics based on a morphology criterion
    # (default: number of connected components per slice)
    fm_id_path = run_heuristics(project_folder)

    # run annotations to quickly filter for the false merges that need to be
    # corrected. this is not strictly necessary (we can use run_correction directly)
    # but in my experience, it is faster to first just filter for false merges
    # and then to correct them.
    run_annotations(fm_id_path, project_folder)
    annotation_path = os.path.join(project_folder, 'annotations.json')
    fm_id_path = filter_fm_ids_by_annotations(project_folder,
                                              fm_id_path, annotation_path)

    # run the false merge correction tool
    run_correction(project_folder, fm_id_path)

    # export the corrected node labels to paintera
    export_correction(project_folder)


if __name__ == '__main__':
    correction_workflow_from_ranked_false_merges()
