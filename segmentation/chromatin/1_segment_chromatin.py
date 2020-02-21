import os
import argparse
from pybdv.metadata import get_data_path
from mmpb.segmentatio.chromatin import chromatin_segmentation_workflow

ROOT = '../../data'


# all nuclei should be predicted with these settings but the largest can't undergo the final
# processing to write the ilastik output h5 file to the main file (memory errors)
# for these process them separately with function process_ilastik_output using more memory
# 8 cores, 384 GB memory

# A few label ids near the end failed
# >> these were tiny fragments of usually 6 pixels or less > could easily add
# in a filter for the minimum number of pixels to avoid these causing errors
def segment_chromatin(version, ilastik_project, ilastik_directory):
    version_folder = os.path.join(ROOT, version)
    assert os.path.exists(version_folder), version_folder

    raw_path = os.path.join(version_folder, 'images', 'local', 'sbem-6dpf-1-whole-raw.xml')
    raw_path = get_data_path(raw_path, return_absolute_path=True)
    nucleus_seg_path = os.path.join(version_folder, 'images', 'local',
                                    'sbem-6dpf-1-whole-segmented-nuclei.xml')
    nucleus_seg_path = get_data_path(nucleus_seg_path, return_absolute_path=True)
    nuclei_table = os.path.join(version_folder, 'tables',
                                'sbem-6dpf-1-whole-segmented-nuclei-labels', 'default.csv')

    tmp_input = 'tmp_chromatin_prediction/tmp_input'
    tmp_output = 'tmp_chromatin_prediction/tmp_output'
    os.makedirs(tmp_input, exist_ok=True)
    os.makedirs(tmp_output, exist_ok=True)
    final_output = './chromatin_prediction.h5'

    # in general run on cluster - 256GB ram, 32 cores
    chromatin_segmentation_workflow(nuclei_table, nucleus_seg_path,
                                    ilastik_project, ilastik_directory,
                                    tmp_input, tmp_output,
                                    final_output, raw_path,
                                    chunk_size=3000, cores=32, memory=254000)


if __name__ == '__main__':
    default_ilastik_project = './nuclei_classification.ilp'
    # need to download ilastik from https://www.ilastik.org/download.html
    default_ilastik_directory = '../../software/ilastik-1.3.3-Linux'

    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=str)
    parser.add_argument('--ilastik_project', type=str, default=default_ilastik_project)
    parser.add_argument('--ilastik_directory', type=str, default=default_ilastik_directory)

    args = parser.parse_args()
    segment_chromatin(args.version, args.ilastik_project, args.ilastik_directory)
