import os
import argparse
from elf.io import open_file
from mmpb.segmentation.network.prediction import prediction, prefilter_blocks

ROOT = '../../data'


def predict_cuticle(path, ckpt, target, gpus):
    input_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-raw.n5')
    input_key = 'setup0/timepoint0/s2'
    assert os.path.exists(input_path), input_path

    mask_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-segmented-shell.n5')
    mask_key = 'setup0/timepoint0/s0'
    assert os.path.exists(mask_path), mask_path

    assert os.path.exists(ckpt), ckpt
    with open_file(input_path, 'r') as f:
        shape = f[input_key].shape

    input_blocks = (96, 256, 256)
    output_blocks = (64, 192, 192)
    out_key = {'volumes/cuticle/foreground': (0, 1),
               'volumes/cuticle/affinities': (1, 10)}

    save_file = 'shell_blocks.json'
    block_list = prefilter_blocks(mask_path, mask_key, shape, output_blocks, save_file)
    print("Running prediction for", len(block_list),  "blocks")

    # TODO need to update this so that it works for slurm
    if target == 'local':
        gpu_mapping = {job_id: gpu for job_id, gpu in enumerate(gpus)}
    else:
        assert False, "Need to fix device-mapping for slurm"
    tmp_folder = './tmp_predict_cuticle'

    n_threads = 6
    prediction(input_path, input_key,
               path, out_key,
               ckpt, tmp_folder, gpu_mapping,
               target, input_blocks, output_blocks,
               n_threads=n_threads, block_list=block_list)


if __name__ == '__main__':
    # oritginal checkpoint
    # ckpt = '/g/kreshuk/data/arendt/platyneris_v1/trained_networks/cuticle_V7/Weights'
    # clean retrain
    default_ckpt = './checkpoints/V1/Weights/best_model.nn'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--ckpt', type=str, default=default_ckpt)
    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    predict_cuticle(args.path, args.ckpt, args.target, args.gpus)
