import os
import json
import luigi
from cluster_tools.inference import InferenceLocal, InferenceSlurm


def prediction(input_path, input_key,
               output_path, output_key,
               ckpt, tmp_folder,
               gpu_mapping, target,
               input_blocks, output_blocks,
               mask_path=None, mask_key=None,
               roi_begin=None, roi_end=None):
    task = InferenceLocal if target == 'local' else InferenceSlurm

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    halo = [(ib - ob) // 2 for ib, ob in zip(input_blocks, output_blocks)]

    # TODO get shebang from config
    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/platybrowser-train/bin/python"
    global_config = task.default_global_config()
    global_config.update({'shebang': shebang,
                          'block_shape': output_blocks,
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    config = task.default_task_config()
    config.update({'chunks': [ob // 2 for ob in output_blocks],
                   'mem_limit': 32, 'time_limit': 720,
                   'threads_per_job': 4, 'device_mapping': gpu_mapping})
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
