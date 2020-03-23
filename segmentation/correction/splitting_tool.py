import argparse
import json
import os


def preprocess_for_project(project_folder, tool_project_folder):
    from mmpb.segmentation.correction.preprocess import preprocess_from_paintera_project
    raw_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5'
    raw_root_key = 'setup0/timepoint0'

    # TODO if I want everyone to run this script, need to move the affinities somewhere to the
    # arendt share as well
    affinities_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    affinities_key = 'volumes/affinities/s1'

    out_key = 'volumes/segmentation_before_splitting'

    # TODO for running on VM need to decrease max jobs to 8
    target = 'local'
    max_jobs = 48

    project_id = project_folder[-2:]
    project_id = int(project_id)

    this_path = os.path.abspath(__file__)
    this_path = os.path.split(__file__)[0]
    tmp_folder = os.path.join(this_path, 'tmps/tmp_project%i_splitting' % project_id)
    roi_file = os.path.join(this_path, 'configs/rois.json')

    with open(roi_file) as f:
        rois = json.load(f)
    roi_begin, roi_end = rois[str(project_id)]

    # TODO determine which scale we need
    scale = 2
    preprocess_from_paintera_project(project_folder, tool_project_folder,
                                     raw_path, raw_root_key,
                                     affinities_path, affinities_key,
                                     out_key, scale,
                                     tmp_folder, target, max_jobs,
                                     roi_begin=roi_begin, roi_end=roi_end)


def run_splitting_tool(tool_project_folder):
    from mmpb.segmentation.correction import CorrectionTool
    config_file = os.path.join(tool_project_folder, 'correct_false_merges_config.json')
    assert os.path.exists(config_file), "Could not find %s, something in pre-processing went wrong!" % config_file
    print("Start splitting tool")
    splitter = CorrectionTool(tool_project_folder)
    splitter()


def main(path):
    tool_project_folder = os.path.join(path, 'splitting_tool')
    if os.path.exists(tool_project_folder):
        run_splitting_tool(tool_project_folder)
    else:
        # try catch around this and clean up if something goes wrong ?
        preprocess_for_project(path, tool_project_folder)


def debug(path):
    import numpy as np
    import z5py
    from heimdall import view

    p = os.path.join(path, 'data.n5')

    p_raw = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5'
    k_raw = 'setup0/timepoint0/s1'
    f_raw = z5py.File(p_raw, 'r')
    ds_raw = f_raw[k_raw]
    ds_raw.n_threads = 8

    f_seg = z5py.File(p)
    k_seg = 'volumes/segmentation_before_splitting'
    ds_seg = f_seg[k_seg]
    ds_seg.n_threads = 8
    assert ds_raw.shape == ds_seg.shape, "%s, %s" % (ds_raw.shape, ds_seg.shape)

    proj_id = int(path[-2:])
    this_path = os.path.abspath(__file__)
    this_path = os.path.split(__file__)[0]
    roi_file = os.path.join(this_path, 'configs/rois.json')
    with open(roi_file) as f:
        rois = json.load(f)
    roi_begin, roi_end = rois[str(proj_id)]

    halo = [50, 512, 512]
    center = [rb + (re - rb) // 2 for re, rb in zip(roi_begin, roi_end)]
    print(roi_begin, roi_end)
    print(center)
    bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
    print(bb)

    raw = ds_raw[bb]
    seg = ds_seg[bb]
    view(raw, seg)
    return

    k = 'morphology'
    with z5py.File(p, 'r') as f:
        ds = f[k]
        m = ds[:]
    starts = m[:, 5:8]
    stops = m[:, 8:11]

    print(starts.min(axis=0))
    print(starts.max(axis=0))
    print()
    print(stops.min(axis=0))
    print(stops.max(axis=0))
    return

    seg_root_key = 'volumes/paintera'
    ass_key = os.path.join(seg_root_key, 'fragment-segment-assignment')
    with z5py.File(p, 'r') as f:
        assignments = f[ass_key][:].T
        seg_ids = assignments[:, 1]
    unique_ids = np.unique(seg_ids)
    if unique_ids[0] == 0:
        unique_ids = unique_ids[1:]
    unique_ids = unique_ids

    # print(len(starts))
    # print(unique_ids.max())
    # return

    attrs_p = os.path.join(path, 'attributes.json')
    with open(attrs_p) as f:
        attrs = json.load(f)

    seg_state = attrs['paintera']['sourceInfo']['sources'][1]['state']
    locked_ids = seg_state['lockedSegments']

    flagged_ids = np.array(list(set(unique_ids.tolist()) - set(locked_ids)))
    flagged_ids = flagged_ids[np.isin(flagged_ids, unique_ids)].tolist()

    for flag_id in flagged_ids:
        print(flag_id)
        if flag_id >= len(starts):
            print("%i is out of bounds %i" % (flag_id, len(starts)))
            continue
        print(starts[flag_id])
        print(stops[flag_id])
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--debug_mode', default=0, type=int)
    args = parser.parse_args()

    path = args.path
    assert os.path.exists(path), "Cannot find valid project @ %s" % path
    debug_mode = int(args.debug_mode)

    if debug_mode:
        debug(path)
    else:
        main(path)
