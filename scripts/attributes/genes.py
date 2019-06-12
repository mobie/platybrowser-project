import os
import json
import luigi
import numpy as np
from ..extension.attributes import GenesLocal, GenesSlurm


def write_genes_table(segm_file, genes_file, table_file, labels,
                      tmp_folder, target, n_threads=8):
    task = GenesSlurm if target == 'slurm' else GenesLocal

    seg_dset = 't00000/s00/4/cells'
    # TODO would be good not to hard-code this
    gene_shape = (570, 518, 550)

    config_folder = os.path.join(tmp_folder, 'configs')
    config = task.default_task_config()
    # this is very ram hungry because we load all the genes at once
    config.update({'threads_per_job': n_threads, 'mem_limit': 256})
    with open(os.path.join(config_folder, 'genes.config'), 'w') as f:
        json.dump(config, f)

    # we need to serialize the labels so that they can be loaded by the luigi task
    labels_path = os.path.join(tmp_folder, 'unique_labels.npy')
    np.save(labels_path, labels)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=1,
             segmentation_path=segm_file, segmentation_key=seg_dset,
             genes_path=genes_file, labels_path=labels_path,
             output_path=table_file, gene_shape=gene_shape)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Computing gene expressions failed")
