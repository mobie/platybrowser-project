# initially:
# go over all cilia, load the volume and highlight the cell the cilium was mapped to if applicable

import json
import os
import queue
import sys
import threading

import numpy as np
import pandas as pd
import napari

from heimdall import view, to_source
from elf.io import open_file
from mmpb.files.xml_utils import get_h5_path_from_xml


def xml_to_h5_path(xml_path):
    path = get_h5_path_from_xml(xml_path, return_absolute_path=True)
    return path


class CiliaCorrectionTool:
    n_threads = 1
    queue_len = 2

    def __init__(self, project_folder,
                 version_folder, scale, cilia_cell_table):
        self.project_folder = project_folder
        os.makedirs(self.project_folder, exist_ok=True)
        self.processed_id_file = os.path.join(self.project_folder, 'processed_ids.json')

        assert os.path.exists(version_folder)

        raw_path = os.path.join(version_folder, 'images', 'sbem-6dpf-1-whole-raw.xml')
        self.raw_path = xml_to_h5_path(raw_path)

        cilia_seg_path = os.path.join(version_folder, 'segmentations',
                                      'sbem-6dpf-1-whole-segmented-cilia-labels.xml')
        self.cilia_seg_path = xml_to_h5_path(cilia_seg_path)

        cell_seg_path = os.path.join(version_folder, 'segmentations',
                                     'sbem-6dpf-1-whole-segmented-cells-labels.xml')
        self.cell_seg_path = xml_to_h5_path(cell_seg_path)

        self.cilia_table_path = os.path.join(version_folder, 'tables',
                                             'sbem-6dpf-1-whole-segmented-cilia-labels', 'default.csv')
        self.cilia_cell_table = pd.read_csv(cilia_cell_table, sep='\t')

        self.scale = scale
        self.init_data()
        self.init_queue_and_workers()

    def init_data(self):
        # init the bounding boxes
        table = pd.read_csv(self.cilia_table_path, sep='\t')
        bb_start = table[['bb_min_z', 'bb_min_y', 'bb_min_x']].values.astype('float32')
        bb_start[np.isinf(bb_start)] = 0
        bb_stop = table[['bb_max_z', 'bb_max_y', 'bb_max_x']].values.astype('float32')

        resolution = [.025, .01, .01]
        scale_factor = [2 ** max(0, self.scale - 1)] + 2 * [2 ** self.scale]
        resolution = [res * sf for res, sf in zip(resolution, scale_factor)]

        halo = [4, 32, 32]
        self.bbs = [tuple(slice(int(sta / res - ha), int(sto / res + ha))
                          for sta, sto, res, ha in zip(start, stop, resolution, halo))
                    for start, stop in zip(bb_start, bb_stop)]

        # mapping from cilia ids to seg_ids
        cil_map_ids = self.cilia_cell_table['cilia_id'].values
        cell_map_ids = self.cilia_cell_table['cell_id'].values
        self.id_mapping = {cil_id: cell_id for cil_id, cell_id in zip(cil_map_ids, cell_map_ids)}

        # init the relevant ids
        self.cilia_ids = np.arange(len(self.bbs))
        if os.path.exists(self.processed_id_file):
            with open(self.processed_id_file) as f:
                self.processed_id_map = json.load(f)
        else:
            self.processed_id_map = {}
        self.processed_id_map = {int(k): v for k, v in self.processed_id_map.items()}
        self.processed_ids = list(self.processed_id_map.keys())

        already_processed = np.in1d(self.cilia_ids, self.processed_ids)
        missing_ids = self.cilia_ids[~already_processed]

        # fill the queue
        self.next_queue = queue.Queue()
        for mi in missing_ids:
            # if mi in (0, 1):
            #     continue
            self.next_queue.put_nowait(mi)

    def worker_thread(self):
        while not self.next_queue.empty():
            seg_id = self.next_queue.get()
            print("Loading seg", seg_id)
            qitem = self.load_data(seg_id)
            self.queue.put((seg_id, qitem))

    def init_queue_and_workers(self):
        self.queue = queue.Queue(maxsize=self.queue_len)
        for i in range(self.n_threads):
            t = threading.Thread(name='worker-%i' % i, target=self.worker_thread)
            # t.setDaemon(True)
            t.start()
        save_folder = os.path.join(self.project_folder, 'results')
        os.makedirs(save_folder, exist_ok=True)

    def load_data(self, cil_id):
        if cil_id in (0, 1):
            return None
        cell_seg_key = 't00000/s00/%i/cells' % (self.scale - 1,)
        cil_seg_key = 't00000/s00/%i/cells' % (self.scale + 1,)
        raw_key = 't00000/s00/%i/cells' % self.scale

        bb = self.bbs[cil_id]
        with open_file(self.raw_path, 'r') as f:
            ds = f[raw_key]
            raw = ds[bb]

        with open_file(self.cilia_seg_path, 'r') as f:
            ds = f[cil_seg_key]
            cil_seg = ds[bb].astype('uint32')
        cil_mask = cil_seg == cil_id
        cil_mask = 2 * cil_mask.astype('uint32')

        cell_id = self.id_mapping[cil_id]
        if cell_id in (0, np.nan):
            cell_seg = None
        else:

            with open_file(self.cell_seg_path, 'r') as f:
                ds = f[cell_seg_key]
                cell_seg = ds[bb].astype('uint32')
                cell_seg = (cell_seg == cell_id).astype('uint32')

        return raw, cil_seg, cil_mask, cell_seg

    def __call__(self):
        left_to_process = len(self.cilia_ids) - len(self.processed_ids)
        print("Left to process:", left_to_process)
        while left_to_process > 0:
            seg_id, qitem = self.queue.get()
            self.correct_segment(seg_id, qitem)
            left_to_process = len(self.cilia_ids) - len(self.processed_ids)

    def correct_segment(self, seg_id, qitem):
        if qitem is None:
            return

        print("Processing cilia:", seg_id)
        raw, cil_seg, cil_mask, cell_seg = qitem

        with napari.gui_qt():
            if cell_seg is None:
                viewer = view(to_source(raw, name='raw'), to_source(cil_seg, name='cilia-segmentation'),
                              to_source(cil_mask, name='cilia-mask'), return_viewer=True)
            else:
                viewer = view(to_source(raw, name='raw'), to_source(cil_seg, name='cilia-segmentation'),
                              to_source(cil_mask, name='cilia-mask'), to_source(cell_seg, name='cell-segmentation'),
                              return_viewer=True)

            @viewer.bind_key('c')
            def confirm(viewer):
                print("Confirming the current id", seg_id, "as correct")
                self.processed_id_map[int(seg_id)] = 'correct'

            @viewer.bind_key('b')
            def background(viewer):
                print("Confirming the current id", seg_id, "into background")
                self.processed_id_map[int(seg_id)] = 'background'

            @viewer.bind_key('m')
            def merge(viewer):
                print("Merging the current id", seg_id, "with other cilia")
                valid_input = False
                while not valid_input:
                    merge_id = input("Please enter the merge id:")
                    try:
                        merge_id = int(merge_id)
                        valid_input = True
                    except ValueError:
                        valid_input = False
                        print("You have entered an invalid input", merge_id, "please try again")
                self.processed_id_map[int(seg_id)] = merge_id

            @viewer.bind_key('r')
            def revisit(viewer):
                print("Marking the current id", seg_id, "to be revisited because something is off")
                self.processed_id_map[int(seg_id)] = 'revisit'

            @viewer.bind_key('h')
            def print_help(viewer):
                print("[c] - confirm cilia as correct")
                print("[b] - mark cilia as background")
                print("[m] - merge cilia with other cilia id")
                print("[d] - revisit this cilia")
                print("[q] - quit")

            # save progress and sys.exit
            @viewer.bind_key('q')
            def quit(viewer):
                print("Quit correction tool")
                self.save_result(seg_id)
                sys.exit(0)

        # save the results for this segment
        self.save_result(seg_id)

    def save_result(self, seg_id):
        self.processed_ids.append(seg_id)
        with open(self.processed_id_file, 'w') as f:
            json.dump(self.processed_id_map, f)
