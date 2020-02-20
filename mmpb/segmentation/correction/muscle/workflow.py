import os
import h5py
import z5py
import pandas as pd
import napari
from heimdall import view, to_source
from elf.wrapper.resized_volume import ResizedVolume


def scale_to_res(scale):
    res = {1: [.025, .02, .02],
           2: [.05, .04, .04],
           3: [.1, .08, .08],
           4: [.2, .16, .16],
           5: [.4, .32, .32]}
    return res[scale]


def get_bb(table, lid, res):
    row = table.loc[lid]
    bb_min = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
    bb_max = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
    return tuple(slice(int(mi / re), int(ma / re)) for mi, ma, re in zip(bb_min, bb_max, res))


def view_candidate(raw, mask, muscle):
    save_id, false_merge, save_state, done = False, False, False, False
    with napari.gui_qt():
        viewer = view(to_source(raw, name='raw'),
                      to_source(mask, name='prediction'),
                      to_source(muscle, name='muscle-segmentation'),
                      return_viewer=True)

        # add key bindings
        @viewer.bind_key('y')
        def confirm_id(viewer):
            print("Confirm id requested")
            nonlocal save_id
            save_id = True

        @viewer.bind_key('f')
        def add_false_merge(viewer):
            print("False merge requested")
            nonlocal false_merge
            false_merge = True

        @viewer.bind_key('s')
        def save(viewer):
            print("Save state requested")
            nonlocal save_state
            save_state = True

        @viewer.bind_key('q')
        def quit_(viewer):
            print("Quit requested")
            nonlocal done
            done = True

    return save_id, false_merge, save_state, done


def check_ids(remaining_ids, saved_ids, false_merges, project_path, state):
    scale = 3
    pathr = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    paths = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.3.1',
                         'segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5')
    table_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.3.1',
                              'tables/sbem-6dpf-1-whole-segmented-cells-labels/default.csv')

    table = pd.read_csv(table_path, sep='\t')
    res = scale_to_res(scale)

    fr = z5py.File(pathr, 'r')
    dsr = fr['volumes/raw/s%i' % scale]
    km = 'volumes/labels/muscle'
    dsm = fr[km]
    dsm = ResizedVolume(dsm, shape=dsr.shape)
    assert dsm.shape == dsr.shape

    check_fps, current_id = state['check_fps'], state['current_id']

    with h5py.File(paths, 'r') as fs:
        dss = fs['t00000/s00/%i/cells' % (scale - 1,)]

        for ii, fid in enumerate(remaining_ids):
            bb = get_bb(table, fid, res)
            if check_fps:
                print("Checking false positives - id:", fid)
            else:
                print("Checking false negatives - id:", fid)
            raw = dsr[bb]
            seg = dss[bb]
            muscle = dsm[bb]
            muscle = (muscle > 0).astype('uint32')
            mask = (seg == fid).astype('uint32')
            save_id, false_merge, save_state, done = view_candidate(raw, mask, muscle)

            if save_id:
                saved_ids.append(fid)
                print("Confirm id", fid, "we now have", len(saved_ids), "confirmed ids.")

            if false_merge:
                print("Add id", fid, "to false merges")
                false_merges.append(fid)

            if save_state:
                print("Save current state to", project_path)
                with h5py.File(project_path) as f:
                    f.attrs['check_fps'] = check_fps

                    if 'false_merges' in f:
                        del f['false_merges']
                    if len(false_merges) > 0:
                        f.create_dataset('false_merges', data=false_merges)

                    g = f['false_positives'] if check_fps else f['false_negatives']
                    g.attrs['current_id'] = current_id + ii + 1
                    if 'proofread' in g:
                        del g['proofread']
                    if len(saved_ids) > 0:
                        g.create_dataset('proofread', data=saved_ids)

            if done:
                print("Quit")
                return False
    return True


def load_state(g):
    current_id = g.attrs.get('current_id', 0)
    remaining_ids = g['prediction'][current_id:]
    saved_ids = g['proofread'][:].tolist() if 'proofread' in g else []
    return remaining_ids, saved_ids, current_id


def load_false_merges(f):
    false_merges = f['false_merges'][:].tolist() if 'false_merges' in f else []
    return false_merges


def run_workflow(project_path):
    print("Start  muscle proofreading workflow from", project_path)
    with h5py.File(project_path, 'r') as f:
        attrs = f.attrs
        check_fps = attrs.get('check_fps', True)
        false_merges = load_false_merges(f)
        g = f['false_positives'] if check_fps else f['false_negatives']
        remaining_ids, saved_ids, current_id = load_state(g)

    state = {'check_fps': check_fps, 'current_id': current_id}
    print("Continue workflow for", "false positives" if check_fps else "false negatives", "from id", current_id)
    done = check_ids(remaining_ids, saved_ids, false_merges, project_path, state)

    if check_fps and done:
        with h5py.File(project_path, 'r') as f:
            g = f['false_negatives']
            remaining_ids, saved_ids, current_id = load_state(g)
        state = {'check_fps': False, 'current_id': current_id}
        print("Start workflow for false negatives from id", current_id)
        check_ids(remaining_ids, saved_ids, false_merges, project_path, state)
