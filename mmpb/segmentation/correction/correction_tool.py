import os
import sys
import json
import queue
import threading

import numpy as np
import napari
import nifty
import nifty.distributed as ndist
import nifty.tools as nt
import vigra

from elf.io import open_file
from elf.io.label_multiset_wrapper import LabelMultisetWrapper


# correct false merges via lifted multicut (or watershed)
# load raw data, watershed from bounding box for correction id
# load sub-graph for nodes corresponding to this segment
# (this takes long, so preload 5 or so via daemon process)
# add layer for seeds, resolve the segment via lmc (or watershed)
# once happy, store the new ids and move on to the next
class CorrectionTool:
    n_threads = 2
    queue_len = 3

    def __init__(self, project_folder,
                 false_merge_id_path=None,
                 table_path=None, table_key=None, scale_factor=None,
                 raw_path=None, raw_key=None,
                 ws_path=None, ws_key=None,
                 node_label_path=None, node_label_key=None,
                 problem_path=None, graph_key=None, feat_key=None,
                 load_lazy=False):
        #
        self.project_folder = project_folder
        self.config_file = os.path.join(project_folder, 'correct_false_merges_config.json')
        self.processed_ids_file = os.path.join(project_folder, 'processed_ids.json')
        self.bg_ids_file = os.path.join(project_folder, 'background_ids.json')
        self.annotation_path = os.path.join(project_folder, 'annotations.json')

        if os.path.exists(self.config_file):
            self.read_config()
            serialize_config = False
        else:
            assert false_merge_id_path is not None
            self.false_merge_id_path = false_merge_id_path

            assert table_path is not None
            self.table_path = table_path
            assert table_key is not None
            self.table_key = table_key
            assert scale_factor is not None
            self.scale_factor = scale_factor

            assert raw_path is not None
            self.raw_path = raw_path
            assert raw_key is not None
            self.raw_key = raw_key

            assert ws_path is not None
            self.ws_path = ws_path
            assert ws_key is not None
            self.ws_key = ws_key

            assert node_label_path is not None
            self.node_label_path = node_label_path
            assert node_label_key is not None
            self.node_label_key = node_label_key

            assert problem_path is not None
            self.problem_path = problem_path
            assert graph_key is not None
            self.graph_key = graph_key
            assert feat_key is not None
            self.feat_key = feat_key

            serialize_config = True

        # TODO implement lazy loading
        self.load_lazy = load_lazy
        print("Loading graph, weights and node labels ...")
        self.load_all_data()
        print("... done")

        print("Initializing queues ...")
        self.init_queue_and_workers()
        print("... done")

        if serialize_config:
            self.write_config()
        print("Initialization done")

    def read_config(self):
        with open(self.config_file) as f:
            conf = json.load(f)

        self.false_merge_id_path = conf['false_merge_id_path']
        self.table_path = conf['table_path']
        self.table_key = conf['table_key']
        self.scale_factor = conf['scale_factor']

        self.raw_path, self.raw_key = conf['raw_path'], conf['raw_key']
        self.ws_path, self.ws_key = conf['ws_path'], conf['ws_key']
        self.node_label_path, self.node_label_key = conf['node_label_path'], conf['node_label_key']

        self.problem_path = conf['problem_path']
        self.graph_key = conf['graph_key']
        self.feat_key = conf['feat_key']

    def write_config(self):
        os.makedirs(self.project_folder, exist_ok=True)
        conf = {'false_merge_id_path': self.false_merge_id_path,
                'table_path': self.table_path, 'table_key': self.table_key,
                'scale_factor': self.scale_factor,
                'raw_path': self.raw_path, 'raw_key': self.raw_key,
                'ws_path': self.ws_path, 'ws_key': self.ws_key,
                'node_label_path': self.node_label_path, 'node_label_key': self.node_label_key,
                'problem_path': self.problem_path, 'graph_key': self.graph_key,
                'feat_key': self.feat_key}
        with open(self.config_file, 'w') as f:
            json.dump(conf, f)

    def load_all_data(self):
        self.ds_raw = open_file(self.raw_path)[self.raw_key]
        self.ds_ws = open_file(self.ws_path)[self.ws_key]
        if self.ds_ws.attrs.get('isLabelMultiset', False):
            self.ds_ws = LabelMultisetWrapper(self.ds_ws)

        self.shape = self.ds_raw.shape
        assert self.ds_ws.shape == self.shape

        with open_file(self.node_label_path, 'r') as f:
            self.node_labels = f[self.node_label_key][:]

        with open(self.false_merge_id_path) as f:
            self.false_merge_ids = np.array(json.load(f))

        if os.path.exists(self.processed_ids_file):
            with open(self.processed_ids_file) as f:
                self.processed_ids = json.load(f)
        else:
            self.processed_ids = []

        if os.path.exists(self.bg_ids_file):
            with open(self.bg_ids_file) as f:
                self.background_ids = json.load(f)
        else:
            self.background_ids = []

        already_processed = np.in1d(self.false_merge_ids, self.processed_ids)
        missing_ids = self.false_merge_ids[~already_processed]

        if os.path.exists(self.annotation_path):
            with open(self.annotation_path) as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}

        self.next_queue = queue.Queue()
        for mi in missing_ids:
            self.next_queue.put_nowait(mi)

        self.graph = ndist.Graph(self.problem_path, self.graph_key, self.n_threads)
        self.uv_ids = self.graph.uvIds()
        assert len(self.uv_ids) > 0

        with open_file(self.problem_path, 'r') as f:
            ds = f[self.feat_key]
            ds.n_threads = self.n_threads

            self.probs = ds[:, 0]

        # morphology table entries
        # id (1)
        # size (1)
        # com (3)
        # bb-min (3)
        # bb-max (3)
        with open_file(self.table_path, 'r') as f:
            table = f[self.table_key][:]
        self.bb_starts = table[:, 5:8]
        self.bb_stops = table[:, 8:11]
        self.bb_starts /= self.scale_factor
        self.bb_stops /= self.scale_factor

    def load_subgraph(self, node_ids):
        # weird, this sometimes happens ...
        if len(node_ids) == 0:
            return None, None, None

        inner_edges, _ = self.graph.extractSubgraphFromNodes(node_ids, allowInvalidNodes=True)
        assert len(inner_edges) > 0
        nodes_relabeled, max_id, mapping = vigra.analysis.relabelConsecutive(node_ids,
                                                                             start_label=0,
                                                                             keep_zeros=False)
        uv_ids = self.uv_ids[inner_edges]
        uv_ids = nt.takeDict(mapping, uv_ids)

        # get rid of paintera ignore label
        pt_ignore_label = 18446744073709551615
        edge_mask = (uv_ids == pt_ignore_label).sum(axis=1) == 0
        uv_ids = uv_ids[edge_mask]
        if len(uv_ids) == 0:
            return None, None, None

        max_id = int(nodes_relabeled.max())
        assert uv_ids.max() <= max_id

        n_nodes = max_id + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(uv_ids)

        probs = self.probs[inner_edges]
        assert len(probs) == graph.numberOfEdges
        return graph, probs, mapping

    def load_segment(self, seg_id):

        # get bounding box of this segment
        starts, stops = self.bb_starts[seg_id], self.bb_stops[seg_id]
        halo = [2, 2, 2]
        bb = tuple(slice(max(0, int(sta - ha)),
                         min(sh, int(sto + ha)))
                   for sta, sto, ha, sh in zip(starts, stops, halo, self.shape))

        # extract the sub-graph
        node_ids = np.where(self.node_labels == seg_id)[0].astype('uint64')
        graph, probs, mapping = self.load_subgraph(node_ids)
        if graph is None:
            return None

        # load raw and watershed
        raw = self.ds_raw[bb]
        ws = self.ds_ws[bb]

        # make the segment mask
        ws[~np.isin(ws, node_ids)] = 0
        seg_mask = ws > 0

        return raw, ws, seg_mask.astype('uint32'), graph, probs, mapping

    def worker_thread(self):
        while not self.next_queue.empty():
            seg_id = self.next_queue.get()
            print("Loading seg", seg_id)
            qitem = self.load_segment(seg_id)

            # skip invalid items
            if qitem is None:
                print("Invalid seg id:", seg_id)
                print("skipping it")
                self.processed_ids.append(int(seg_id))
                with open(self.processed_ids_file, 'w') as f:
                    json.dump(self.processed_ids, f)
                continue

            self.queue.put((seg_id, qitem))

    def init_queue_and_workers(self):
        self.queue = queue.Queue(maxsize=self.queue_len)
        for i in range(self.n_threads):
            t = threading.Thread(name='worker-%i' % i, target=self.worker_thread)
            t.setDaemon(True)
            t.start()
        save_folder = os.path.join(self.project_folder, 'results')
        os.makedirs(save_folder, exist_ok=True)

    @staticmethod
    def graph_watershed(graph, probs, ws, seed_points, mapping):
        seed_ids = np.unique(seed_points)[1:]
        if len(seed_ids) == 0:
            return None

        seeds = np.zeros(graph.numberOfNodes, dtype='uint64')
        # TODO this is what takes a long time for large volumes I guess
        # should speed it up
        for seed_id in seed_ids:
            mask = seed_points == seed_id
            seed_nodes = np.unique(ws[mask])
            if seed_nodes[0] == 0:
                seed_nodes = seed_nodes[1:]
            seed_nodes = nt.takeDict(mapping, seed_nodes)
            seeds[seed_nodes] = seed_id

        node_labels = nifty.graph.edgeWeightedWatershedsSegmentation(graph, seeds, probs)
        return node_labels

    def correct_segment(self, seg_id, qitem):
        print("Processing segment:", seg_id)
        raw, ws, seg, graph, probs, mapping = qitem
        ws_ids = np.unique(ws)[1:]
        node_labels = np.ones(graph.numberOfNodes, dtype='uint64')

        seeds = np.zeros_like(ws, dtype='uint32')
        skip_this_segment = False
        is_background = False
        quit_ = False

        with napari.gui_qt():
            viewer = napari.Viewer(title="Segment%i" % seg_id)
            viewer.add_image(raw, name='raw')
            viewer.add_labels(ws, name='ws', visible=False)
            viewer.add_labels(seg, name='seg')
            viewer.add_labels(seeds, name='seeds')

            @viewer.bind_key('h')
            def print_help(viewer):
                print("Put seeds by selecting the 'seeds' layer and painting with different ids")
                print("[w] - run watershed from seeds")
                print("[s] - skip the current segment (if it's not a merge)")
                print("[c] - clear seeds")
                print("[x] - save current data for debugging")
                print("[b] - merge this segment into background")
                print("[q] - quit")

            @viewer.bind_key('s')
            def skip(viewer):
                print("Skipping the current segment")
                nonlocal skip_this_segment
                skip_this_segment = True

            @viewer.bind_key('b')
            def to_background(viewer):
                print("Setting the current segment to backroung")
                nonlocal is_background
                is_background = True

            @viewer.bind_key('x')
            def save_for_debug(viewer):
                print("Saving debug data ...")
                debug_folder = os.path.join(self.project_folder, 'debug')
                os.makedirs(debug_folder, exist_ok=True)

                layers = viewer.layers
                seed_points = layers['seeds'].data
                with open_file(os.path.join(debug_folder, 'data.n5')) as f:
                    f.create_dataset('raw', data=raw, compression='gzip')
                    f.create_dataset('ws', data=ws, compression='gzip')
                    f.create_dataset('seeds', data=seed_points, compression='gzip')

                with open(os.path.join(debug_folder, 'mapping.json'), 'w') as f:
                    json.dump(mapping, f)
                np.save(os.path.join(debug_folder, 'graph.npy'), graph.uvIds())
                np.save(os.path.join(debug_folder, 'probs.npy'), probs)
                print("... done")

            @viewer.bind_key('w')
            def watershed(viewer):
                nonlocal node_labels, seeds

                print("Run watershed from seed layer ...")
                layers = viewer.layers
                ws = layers['ws'].data
                seeds = layers['seeds'].data

                new_node_labels = self.graph_watershed(graph, probs, ws, seeds, mapping)
                if new_node_labels is None:
                    print("Did not find any seeds, doing nothing")
                    return
                else:
                    node_labels = new_node_labels

                label_dict = {wsid: node_labels[mapping[wsid]] for wsid in ws_ids}
                label_dict[0] = 0
                seg = nt.takeDict(label_dict, ws)

                layers['seg'].data = seg
                print("... done")

            @viewer.bind_key('c')
            def clear(viewer):
                nonlocal node_labels, seeds
                print("Clear seeds ...")
                confirm = input("Do you really want to clean the seeds? y / [n]")
                if confirm != 'y':
                    return
                node_labels = np.ones(graph.numberOfNodes, dtype='uint64')
                seeds = np.zeros_like(ws, dtype='uint32')
                viewer.layers['seeds'].data = seeds
                seg = (ws > 0).astype('uint32')
                viewer.layers['seg'].data = seg
                print("... done")

            # save progress and quit()
            @viewer.bind_key('q')
            def quit(viewer):
                nonlocal quit_
                print("Quit correction tool")
                quit_ = True

        # save the results for this segment
        self.save_segment_result(seg_id, ws, seeds, node_labels, mapping,
                                 skip_this_segment, is_background)
        if quit_:
            sys.exit(0)

    def save_segment_result(self, seg_id, ws, seeds, node_labels, mapping, skip, is_background):
        if not (skip or is_background):
            save_file = os.path.join(self.project_folder, 'results', '%i.npz' % seg_id)
            node_ids = list(mapping.keys())
            save_labels = [node_labels[mapping[nid]] for nid in node_ids]

            seed_ids = np.unique(seeds[1:])
            seeded_ids = []
            seed_labels = []
            for seed_id in seed_ids:
                mask = seeds == seed_id
                this_ids = np.unique(ws[mask])
                if this_ids[0] == 0:
                    this_ids = this_ids[1:]
                seeded_ids.extend(this_ids)
                seed_labels.extend(len(this_ids) * [seed_id])

            np.savez_compressed(save_file, node_ids=node_ids, node_labels=save_labels,
                                seeded_ids=seeded_ids, seed_labels=seed_labels)

        self.processed_ids.append(int(seg_id))
        with open(self.processed_ids_file, 'w') as f:
            json.dump(self.processed_ids, f)

        if is_background:
            self.background_ids.append(int(seg_id))
            with open(self.bg_ids_file, 'w') as f:
                json.dump(self.background_ids, f)

        print("Saved results for segment", seg_id)
        print("Processed", len(self.processed_ids), "/", len(self.false_merge_ids), "objects")

    def __call__(self):
        left_to_process = len(self.false_merge_ids) - len(self.processed_ids)
        while left_to_process > 0:
            seg_id, qitem = self.queue.get()
            self.correct_segment(seg_id, qitem)
            left_to_process = len(self.false_merge_ids) - len(self.processed_ids)
        return left_to_process == 0

    @staticmethod
    def debug(debug_folder, n_threads=8):
        with open_file(os.path.join(debug_folder, 'data.n5')) as f:
            ds = f['raw']
            ds.n_threads = n_threads
            raw = ds[:]

            ds = f['ws']
            ds.n_threads = n_threads
            ws = ds[:]

            ds = f['seeds']
            ds.n_threads = n_threads
            seed_points = ds[:]

        with open(os.path.join(debug_folder, 'mapping.json'), 'r') as f:
            mapping = json.load(f)
        mapping = {int(k): v for k, v in mapping.items()}

        uv_ids = np.load(os.path.join(debug_folder, 'graph.npy'))
        n_nodes = int(uv_ids.max()) + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(uv_ids)
        probs = np.load(os.path.join(debug_folder, 'probs.npy'))

        node_labels = CorrectionTool.graph_watershed(graph, probs, ws, seed_points, mapping)

        ws_ids = np.unique(ws)[1:]
        label_dict = {wsid: node_labels[mapping[wsid]] for wsid in ws_ids}
        label_dict[0] = 0
        seg = nt.takeDict(label_dict, ws)

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw, name='raw')
            viewer.add_labels(ws, name='ws')
            viewer.add_labels(seg, name='seg')
