import os
import json
import sys
import luigi
import z5py
from cluster_tools.downscaling import DownscalingWorkflow


def preprocess():
    path = './data.n5'
    in_key = 'labels_for_subdivision_mip/s0'
    root_key = 'labels_for_subdivision_mip'

    n_scales = 4
    scale_factors = n_scales * [[2, 2, 2]]
    halos = scale_factors

    tmp_folder = './tmp_subdivision_labels/tmp2'
    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    conf = DownscalingWorkflow.get_config()['downscaling']
    conf.update({'library_kwargs': {'order': 0}})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(conf, f)

    target = 'local'
    max_jobs = 16

    task = DownscalingWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                               target=target, max_jobs=max_jobs,
                               input_path=path, input_key=in_key,
                               scale_factors=scale_factors, halos=halos,
                               output_path=path, output_key_prefix=root_key)
    luigi.build([task], local_scheduler=True)


def find_my_block(block_id):
    from heimdall import view, to_source

    scale = 5
    rpath = '../../../data/rawdata/sbem-6dpf-1-whole-raw.n5'
    k = 'setup0/timepoint0/s%i' % scale

    f = z5py.File(rpath)
    ds = f[k]
    ds.n_thread = 8
    raw = ds[:]

    scale_blocks = scale - 3
    path = './data.n5'
    k = 'labels_for_subdivision_mip/s%i' % scale_blocks
    f = z5py.File(path)
    ds = f[k]
    ds.n_threads = 8
    block = ds[:]
    block = (block == block_id).astype('uint32')

    view(to_source(raw, name='raw'),
         to_source(block, name='block-volume'))


if __name__ == '__main__':
    # preprocess()
    block_id = int(sys.argv[1])
    find_my_block(block_id)
