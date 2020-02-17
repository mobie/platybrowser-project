import os
import argparse
from mmpb.segmentation.network import prediction

ROOT = '../../data'


def predict_cells(output_path, ckpt, target, gpus):
    # block-shapes
    # remove (15, 30, 30) pixels from each side in the output
    input_path = os.path.join(ROOT, 'rawdata/sbem-6dpf-1-whole-raw.n5')
    input_key = 'setup0/timepoint0/s1'

    mask_path = os.path.join(ROOT, 'sbem-6dpf-1-whole-segmented-inside.n5')
    mask_key = 'setup0/timepoint0/s0'

    input_blocks = (90, 270, 270)
    output_blocks = (60, 210, 210)
    output_key = {'volumes/affinities/s1': (0, 3)}

    # TODO gpus -> gpu mapping
    gpu_mapping = gpus
    tmp_folder = './tmp_predict_cells'

    roi_begin = roi_end = None
    prediction(input_path, input_key,
               output_path, output_key,
               ckpt, tmp_folder, gpu_mapping,
               target, input_blocks, output_blocks,
               mask_path=mask_path, mask_key=mask_key,
               roi_begin=roi_begin, roi_end=roi_end)


if __name__ == '__main__':
    # TODO update
    default_ckpt = '/g/kreshuk/data/arendt/platyneris_v1/trained_networks/unet_lr_v5/Weights'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='../data.n5')
    parser.add_argument('--ckpt', type=str, default=default_ckpt)
    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    predict_cells(args.output_path, args.ckpt, args.target, args.gpus)
