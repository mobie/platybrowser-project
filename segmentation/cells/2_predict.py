import os
import argparse
from elf.io import open_file
from mmpb.segmentation.network.prediction import prediction

ROOT = '../../data'


def get_roi(path, key, halo=[100, 1024, 1024]):
    with open_file(path, 'r') as f:
        shape = f[key].shape
    roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    return roi_begin, roi_end


def predict_cells(path, ckpt, target, gpus, with_roi=False):
    input_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-raw.n5')
    input_key = 'setup0/timepoint0/s1'
    assert os.path.exists(input_path), input_path

    mask_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-segmented-inside.n5')
    mask_key = 'setup0/timepoint0/s0'
    assert os.path.exists(mask_path)

    # block-shapes:
    # remove (16, 32, 32) pixels from each side in the output
    input_blocks = (96, 256, 256)
    output_blocks = (64, 192, 192)
    output_key = {'volumes/affinities/s1': (0, 3)}

    # TODO need to update this so that it works for slurm
    if target == 'local':
        gpu_mapping = {job_id: gpu for job_id, gpu in enumerate(gpus)}
    else:
        assert False, "Need to fix device-mapping for slurm"
    tmp_folder = './tmp_predict_cells'

    if with_roi:
        roi_begin, roi_end = get_roi(input_path, input_key)
        print("Have bounding box", roi_begin, "to", roi_end)
    else:
        roi_begin = roi_end = None

    prediction(input_path, input_key,
               path, output_key,
               ckpt, tmp_folder, gpu_mapping,
               target, input_blocks, output_blocks,
               mask_path=mask_path, mask_key=mask_key,
               roi_begin=roi_begin, roi_end=roi_end)


if __name__ == '__main__':
    # oritginal checkpoint
    # ckpt = '/g/kreshuk/data/arendt/platyneris_v1/trained_networks/unet_lr_v5'
    # clean retrain
    default_ckpt = './checkpoints/V1/Weights/best_model.nn'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data.n5')
    parser.add_argument('--ckpt', type=str, default=default_ckpt)
    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--with_roi', type=int, default=0)
    args = parser.parse_args()

    predict_cells(args.path, args.ckpt, args.target, args.gpus, bool(args.with_roi))
