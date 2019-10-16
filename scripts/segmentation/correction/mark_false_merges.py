import os
import json
import napari
from elf.io import open_file


# TODO
# mark false merges in segmentation:
# open napari viewer with raw data and segmentation (in appropriate res)
# set annotations in 'has_false_merge' and 'is_correct' layer
# clear segmentations that have an annotation (upon key press)
# store annotations in the project folder
class MarkFalseMerges:
    def __init__(self, project_folder,
                 raw_path=None, raw_key=None,
                 seg_path=None, seg_key=None):
        pass

    def __call__(self):
        pass
