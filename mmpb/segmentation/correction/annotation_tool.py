import os
import json
import queue
import sys
import threading

import numpy as np
import napari

from elf.io import open_file


# Annotate segments
class AnnotationTool:
    n_threads = 3
    queue_len = 6

    def __init__(self, project_folder, id_path,
                 table_path, table_key, scale_factor,
                 raw_path, raw_key,
                 ws_path, ws_key,
                 node_label_path=None,
                 node_label_key=None):
        #
        self.project_folder = project_folder
        os.makedirs(self.project_folder, exist_ok=True)
        self.processed_ids_file = os.path.join(project_folder, 'processed_ids.json')
        self.annotation_path = os.path.join(project_folder, 'annotations.json')

        self.id_path = id_path

        self.table_path = table_path
        self.table_key = table_key
        self.scale_factor = scale_factor

        self.raw_path = raw_path
        self.raw_key = raw_key

        self.ws_path = ws_path
        self.ws_key = ws_key

        self.node_label_path = node_label_path
        self.node_label_key = node_label_key

        self.load_all_data()
        self.init_queue_and_workers()
        print("Initialization done")

    def load_all_data(self):
        self.ds_raw = open_file(self.raw_path)[self.raw_key]
        self.ds_ws = open_file(self.ws_path)[self.ws_key]
        self.shape = self.ds_raw.shape
        assert self.ds_ws.shape == self.shape

        if self.node_label_path is None:
            self.node_labels = None
        else:
            with open_file(self.node_label_path, 'r') as f:
                self.node_labels = f[self.node_label_key][:]

        with open(self.id_path) as f:
            self.ids = np.array(json.load(f))

        if os.path.exists(self.processed_ids_file):
            with open(self.processed_ids_file) as f:
                self.processed_ids = json.load(f)
        else:
            self.processed_ids = []

        already_processed = np.in1d(self.ids, self.processed_ids)
        missing_ids = self.ids[~already_processed]

        if os.path.exists(self.annotation_path):
            with open(self.annotation_path) as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}

        self.next_queue = queue.Queue()
        for mi in missing_ids:
            self.next_queue.put_nowait(mi)

        # morphology table entries
        # id (1)
        # size (1)
        # com (3)
        # bb-min (3)
        # bb-max (3)
        with open_file(self.table_path, 'r') as f:
            table = f[self.table_key][:]
        self.bb_starts = table[:, 5:8]
        self.bb_stops = table[:, 8:]
        self.bb_starts /= self.scale_factor
        self.bb_stops /= self.scale_factor

    def load_segment(self, seg_id):

        # get bounding box of this segment
        starts, stops = self.bb_starts[seg_id], self.bb_stops[seg_id]
        halo = [2, 2, 2]
        bb = tuple(slice(max(0, int(sta - ha)),
                         min(sh, int(sto + ha)))
                   for sta, sto, ha, sh in zip(starts, stops, halo, self.shape))

        # load raw and watershed
        raw = self.ds_raw[bb]
        ws = self.ds_ws[bb]

        # make the segment mask
        if self.node_labels is None:
            seg_mask = (ws == seg_id).astype('uint32')
            ws = None
        else:
            node_ids = np.where(self.node_labels == seg_id)[0].astype('uint64')
            seg_mask = np.isin(ws, node_ids)
            ws[~seg_mask] = 0

        return raw, ws, seg_mask.astype('uint32')

    def worker_thread(self):
        while not self.next_queue.empty():
            seg_id = self.next_queue.get()
            print("Loading seg", seg_id)
            qitem = self.load_segment(seg_id)
            self.queue.put((seg_id, qitem))

    def init_queue_and_workers(self):
        self.queue = queue.Queue(maxsize=self.queue_len)
        for i in range(self.n_threads):
            t = threading.Thread(name='worker-%i' % i, target=self.worker_thread)
            t.setDaemon(True)
            t.start()

    def correct_segment(self, seg_id, qitem):
        print("Processing segment:", seg_id)
        raw, ws, seg = qitem
        annotation = None

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw, name='raw')
            if ws is not None:
                viewer.add_labels(ws, name='ws')
            viewer.add_labels(seg, name='seg')

            @viewer.bind_key('h')
            def print_help(viewer):
                print("The following annotations are available:")
                print("[r] - needs to be revisited")
                print("[i] - incomplete, needs to be merged with other id")
                print("[m] - merge with background")
                print("[y] - confirm segment")
                print("[c] - enter custom annotation")
                print("Other options:")
                print("[q] - quit")

            @viewer.bind_key('r')
            def revisit(viewer):
                nonlocal annotation
                print("Setting annotation to revisit")
                annotation = 'revisit'

            @viewer.bind_key('m')
            def merge(viewer):
                nonlocal annotation
                print("Setting annotation to merge")
                annotation = 'merge'

            @viewer.bind_key('i')
            def incomplete(viewer):
                nonlocal annotation
                print("Setting annotation to incomplete")
                annotation = 'incomplete'

            @viewer.bind_key('y')
            def confirm(viewer):
                nonlocal annotation
                print("Setting annotation to confirm")
                annotation = 'confirm'

            @viewer.bind_key('c')
            def custom(viewer):
                nonlocal annotation
                annotation = input("Enter custom annotation")

            @viewer.bind_key('q')
            def quit(viewer):
                self.save_annotation(seg_id, annotation)
                sys.exit(0)

        self.save_annotation(seg_id, annotation)

    def save_annotation(self, seg_id, annotation):
        self.processed_ids.append(int(seg_id))
        with open(self.processed_ids_file, 'w') as f:
            json.dump(self.processed_ids, f)
        if annotation is None:
            return
        self.annotations[int(seg_id)] = annotation
        with open(self.annotation_path, 'w') as f:
            json.dump(self.annotations, f)

    def __call__(self):
        left_to_process = len(self.ids) - len(self.processed_ids)
        while left_to_process > 0:
            seg_id, qitem = self.queue.get()
            self.correct_segment(seg_id, qitem)
            left_to_process = len(self.ids) - len(self.processed_ids)
