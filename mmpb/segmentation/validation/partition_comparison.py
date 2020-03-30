import os
import json
import luigi
from cluster_tools.evaluation import EvaluationWorkflow
from ..default_config import write_default_global_config


def partition_comparison(seg_path, seg_key, ref_path, ref_key,
                         tmp_folder, target, max_jobs,
                         roi_begin=None, roi_end=None):
    task = EvaluationWorkflow

    out_path = os.path.join(tmp_folder, 'results.json')
    config_dir = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_dir, roi_begin, roi_end)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Evaluation workflow failed.")

    with open(out_path) as f:
        result = json.load(f)

    return result
