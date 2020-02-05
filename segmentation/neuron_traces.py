#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser/bin/python
import json
import pandas as pd
from mmpb.export import extract_neuron_traces


# for debugging
def check_extraction():
    import elf.skeleton.io as skio
    path = '/g/kreshuk/data/arendt/platyneris_v1/tracings/kevin/knm_ApNS_6dpf_neuron_traces.4138.nmx'
    # path = '/g/kreshuk/data/arendt/platyneris_v1/tracings/comm_sec_seg.019.nmx'
    skel = skio.read_nml(path)
    search_str = 'neuron_id'
    for k, v in skel.items():
        sub = k.find(search_str)
        beg = sub + len(search_str)
        end = k.find('.', beg)
        n_id = int(k[beg:end])

        if n_id != 10:
            continue

        print(k)
        print(n_id)
        print(v)


def export_traces():
    folder = '/g/kreshuk/data/arendt/platyneris_v1/tracings/kevin'
    ref_path = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    seg_out_path = './sbem-6dpf-1-whole-traces.xml'
    table_out_path = './sbem-6dpf-1-whole-traces-table-default.csv'
    tmp_folder = './tmp_traces'

    cell_seg_info = {'path': '../data/0.3.1/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5',
                     'scale': 2}
    nucleus_seg_info = {'path': '../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5',
                        'scale': 0}

    extract_neuron_traces(folder, ref_path, seg_out_path, table_out_path, tmp_folder,
                          cell_seg_info, nucleus_seg_info)


def get_cell_ids():
    table_path = './sbem-6dpf-1-whole-traces-table-default.csv'
    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id'].values
    cell_ids = cell_ids[cell_ids != 0].tolist()
    with open('./trace_cell_ids.json', 'w') as f:
        json.dump(cell_ids, f)


if __name__ == '__main__':
    export_traces()
    get_cell_ids()
    # check_extraction()
