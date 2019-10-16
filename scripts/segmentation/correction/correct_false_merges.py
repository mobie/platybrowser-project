import os
import napari
from elf.io import open_file


# TODO
# correct false merges via lifted multicut (or watershed)
# load raw data, watershed from bounding box for correction id
# load sub-graph for nodes corresponding to this segment
# (this takes long, so preload 5 or so via daemon process)
# add layer for seeds, resolve the segment via lmc (or watershed)
# once happy, store the new ids and move on to the next
class CorrectFalseMerges:
    def __init__(self, project_folder,
                 correct_id_path=None, table_path=None,
                 raw_path=None, raw_key=None,
                 ws_path=None, ws_key=None,
                 node_label_path=None, node_label_key=None,
                 problem_path=None, graph_key=None, cost_key=None):
        pass

    def __call__(self):
        pass
