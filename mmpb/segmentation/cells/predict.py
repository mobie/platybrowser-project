import os
import json
import luigi
from cluster_tools.inference import InferenceLocal, InferenceSlurm


def predict(ckpt, tmp_folder, target, max_jobs):
    # TODO update the paths / keys
    input_path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
    output_path = input_path
    roi_begin = roi_end = None
    in_key = 'volumes/raw/s1'
    out_key = {'volumes/affinities/s1': (0, 3)}
    mask_path = input_path
    mask_key = 'volumes/mask/s5'

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    input_blocks = (90, 270, 270)
    # remove (15, 30, 30) pixels from each side in the output
    output_blocks = (60, 210, 210)
    halo = [(ib - ob) // 2 for ib, ob in zip(input_blocks, output_blocks)]

    # TODO update shebang
    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch/bin/python"
    global_config = InferenceLocal.default_global_config()
    global_config.update({'shebang': shebang,
                          'block_shape': output_blocks,
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    # TODO update the gpu mapping
    config = InferenceLocal.default_task_config()
    config.update({'chunks': [ob // 2 for ob in output_blocks],
                   'mem_limit': 32, 'time_limit': 720,
                   'threads_per_job': 4, 'set_visible_device': False})
    with open(os.path.join(config_folder, 'inference.config'), 'w') as f:
        json.dump(config, f)

    task = InferenceLocal if target == 'local' else InferenceSlurm
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder,
             input_path=input_path, input_key=in_key,
             output_path=output_path, output_key=out_key,
             mask_path=mask_path, mask_key=mask_key,
             checkpoint_path=ckpt, framework='inferno',
             halo=halo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Inference failed"


# TODO move to scripts
if __name__ == '__main__':
    # target = 'local'
    # max_jobs = 1

    target = 'slurm'
    max_jobs = 16

    ckpt = '/g/kreshuk/data/arendt/platyneris_v1/trained_networks/unet_lr_v5/Weights'
    tmp_folder = ''

    # TODO pass the paths
    predict(ckpt, tmp_folder, target, max_jobs)
