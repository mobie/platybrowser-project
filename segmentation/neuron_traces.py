#! /g/arendt/pape/miniconda3/envs/platybrowser/bin/python
from scripts.export import extract_neuron_traces


def export_test_traces():
    folder = '/g/kreshuk/data/arendt/platyneris_v1/tracings'
    ref_path = '../data/rawdata/sbem-6dpf-1-whole-raw.h5'
    seg_out_path = './sbem-6dpf-1-whole-traces.xml'
    table_out_path = './sbem-6dpf-1-whole-traces-table-default.xml'
    tmp_folder = './tmp_traces'
    extract_neuron_traces(folder, ref_path, seg_out_path, table_out_path, tmp_folder)


if __name__ == '__main__':
    export_test_traces()
