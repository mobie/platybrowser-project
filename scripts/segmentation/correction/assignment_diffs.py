import json
import os

import numpy as np
import luigi
import nifty.ground_truth as ngt
from cluster_tools.node_labels import NodeLabelWorkflow

# TODO some of this is general purpose and shoulg go to elf


def node_labels(ws_path, ws_key, input_path, input_key,
                output_path, output_key, prefix,
                tmp_folder, target='slurm', max_jobs=250):
    task = NodeLabelWorkflow

    configs = task.get_config()
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    conf = configs['global']
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python3.7'
    conf['shebang'] = shebang
    conf['block_shape'] = [16, 256, 256]

    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['block_node_labels']
    conf.update({'mem_limit': 4, 'time_limit': 120})
    with open(os.path.join(config_folder, 'block_node_labels.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             ws_path=ws_path, ws_key=ws_key,
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             prefix=prefix)
    luigi.build([t], local_scheduler=True)


# we only look at objects in the reference assignment that are SPLIT in
# the new assignment. For the other direction, just reverse reference and new
def assignment_diff_splits(reference_assignment, new_assignment):
    assert reference_assignment.shape == new_assignment.shape
    reference_ids = np.unique(reference_assignment)

    print("Computing assignment diff ...")
    ovlp_comp = ngt.overlap(reference_assignment, new_assignment)
    split_ids = [ovlp_comp.overlapArrays(ref_id)[0] for ref_id in reference_ids]
    split_ids = {int(ref_id): len(ovlps) for ref_id, ovlps in zip(reference_ids, split_ids)
                 if len(ovlps) > 1}
    return split_ids
