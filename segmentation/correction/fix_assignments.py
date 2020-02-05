import os
import json


def get_assignments(version):
    from mmpb.segmentation.correction.assignment_diffs import node_labels
    assert version in ('0.5.5', '0.6.1', 'local')
    ws_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    ws_key = 'volumes/paintera/proofread_cells_multiset/data/s0'

    if version == 'local':
        in_path = './data.n5'
        in_key = 'segmentation/s0'
        prefix = version
    else:
        in_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data',
                               '%s/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5' % version)
        in_key = 't00000/s00/0/cells'

        prefix = ''.join(version.split('.'))

    out_path = './data.n5'
    out_key = 'assignments/%s' % prefix

    tmp_folder = './tmp_%s' % prefix

    node_labels(ws_path, ws_key, in_path, in_key,
                out_path, out_key, prefix, tmp_folder)


def get_split_assignments():
    import z5py
    from mmpb.segmentation.correction.assignment_diffs import assignment_diff_splits
    with z5py.File('./data.n5') as f:
        ref = f['assignments/055'][:]
        new = f['assignments/corrected'][:]

    splits = assignment_diff_splits(ref, new)

    print(len(splits))

    # with open('./split_assignments_local.json', 'w') as f:
    #     json.dump(splits, f)


def blub():
    with open('./split_assignments.json') as f:
        split_assignments = json.load(f)
    ids = list(split_assignments.keys())
    # vals = list(split_assignments.values())

    correction_path = './proj_correct2/results'
    correct_ids = os.listdir(correction_path)
    correct_ids = [int(os.path.splitext(cid)[0]) for cid in correct_ids]

    print("Number of assignments which are split:", len(ids))
    print("Number of assignments which were supposed to be split:", len(correct_ids))

    diff = set(ids) - set(correct_ids)
    print("Number after diff:", len(diff))


if __name__ == '__main__':
    # get_assignments('0.5.5')
    # get_assignments('0.6.1')
    # get_assignments('local')

    get_split_assignments()
    # blub()
