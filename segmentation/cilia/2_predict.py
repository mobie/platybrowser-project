import os
import argparse
from elf.io import open_file
from mmpb.segmentation.network.prediction import prediction

ROOT = '../../data'


def predict_cells(path, ckpt, target, gpus):
    input_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-raw.n5')
    input_key = 'setup0/timepoint0/s0'
    assert os.path.exists(input_path), input_path

    mask_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-segmented-nephridia.n5')
    mask_key = 'setup0/timepoint0/s0'
    assert os.path.exists(mask_path)

    # block-shapes:
    input_blocks = (96, 288, 288)
    output_blocks = (76, 228, 228)
    out_key = {'volumes/cilia/foreground': (0, 1),
               'volumes/cilia/affinities': (1, 13)}

    # TODO need to update this so that it works for slurm
    if target == 'local':
        gpu_mapping = {job_id: gpu for job_id, gpu in enumerate(gpus)}
    else:
        assert False, "Need to fix device-mapping for slurm"
    tmp_folder = './tmp_predict_cilia'

    prediction(input_path, input_key,
               path, output_key,
               ckpt, tmp_folder, gpu_mapping,
               target, input_blocks, output_blocks,
               mask_path=mask_path, mask_key=mask_key)


if __name__ == '__main__':
    # oritginal checkpoint
    # ckpt = '/g/kreshuk/data/arendt/platyneris_v1/trained_networks/cilia_V2/Weights'
    # clean retrain
    default_ckpt = './checkpoints/V1/Weights/best_model.nn'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--ckpt', type=str, default=default_ckpt)
    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    predict_cells(args.path, args.ckpt, args.target, args.gpus)
