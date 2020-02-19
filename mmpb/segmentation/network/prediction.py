import os
import json
import luigi
from concurrent import futures
from tqdm import tqdm

import nifty.tools as nt
from cluster_tools.inference import InferenceLocal, InferenceSlurm
from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from mmpb.default_config import get_default_shebang


def prediction(input_path, input_key,
               output_path, output_key,
               ckpt, tmp_folder,
               gpu_mapping, target,
               input_blocks, output_blocks,
               mask_path='', mask_key='',
               roi_begin=None, roi_end=None,
               n_threads=4, block_list=None):
    task = InferenceLocal if target == 'local' else InferenceSlurm

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    if block_list is None:
        block_list_path = None
    else:
        block_list_path = os.path.join(tmp_folder, 'blocks.json')
        with open(block_list_path, 'w') as f:
            json.dump(block_list, f)

    halo = [(ib - ob) // 2 for ib, ob in zip(input_blocks, output_blocks)]

    shebang = get_default_shebang()
    global_config = task.default_global_config()
    global_config.update({'shebang': shebang,
                          'block_shape': output_blocks,
                          'roi_begin': roi_begin,
                          'roi_end': roi_end,
                          'block_list_path': block_list_path})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    config = task.default_task_config()
    config.update({'chunks': [ob // 2 for ob in output_blocks],
                   'mem_limit': 32, 'time_limit': 720,
                   'threads_per_job': n_threads, 'device_mapping': gpu_mapping})
    with open(os.path.join(config_folder, 'inference.config'), 'w') as f:
        json.dump(config, f)

    max_jobs = len(gpu_mapping)
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder,
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             mask_path=mask_path, mask_key=mask_key,
             checkpoint_path=ckpt, framework='pytorch',
             halo=halo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Inference failed"


def prefilter_blocks(mask_path, mask_key,
                     shape, block_shape,
                     save_file, n_threads=48):
    if os.path.exists(save_file):
        print("Loading block list from file")
        with open(save_file) as f:
            return json.load(f)

    with open_file(mask_path, 'r') as f:
        ds = f[mask_key]
        mask = ResizedVolume(ds, shape=shape, order=0)

        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        def check_block(block_id):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            d = mask[bb]
            if d.sum() > 0:
                return block_id
            else:
                return None

        print("Computing block list ...")
        with futures.ThreadPoolExecutor(n_threads) as tp:
            blocks = list(tqdm(tp.map(check_block, range(n_blocks)), total=n_blocks))
        blocks = [bid for bid in blocks if bid is not None]

    with open(save_file, 'w') as f:
        json.dump(blocks, f)
    return blocks
