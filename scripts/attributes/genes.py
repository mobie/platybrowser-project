import os
import json
from glob import glob

import luigi
import numpy as np
import h5py
from tqdm import tqdm

from ..extension.attributes import GenesLocal, GenesSlurm
from ..extension.attributes import VCAssignmentsLocal, VCAssignmentsSlurm


def gene_assignment_table(segm_file, genes_file, table_file, labels,
                          tmp_folder, target, n_threads=8):
    task = GenesSlurm if target == 'slurm' else GenesLocal
    seg_dset = 't00000/s00/4/cells'

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
             output_path=table_file)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Computing gene expressions failed")


def vc_assignment_table(seg_path, vc_vol_path, vc_expression_path,
                        med_expression_path, output_path,
                        tmp_folder, target, n_threads=8):
    task = VCAssignmentsSlurm if target == 'slurm' else VCAssignmentsLocal

    config_folder = os.path.join(tmp_folder, 'configs')
    config = task.default_task_config()
    # this is very ram hungry because we load all the genes at once
    config.update({'threads_per_job': n_threads, 'mem_limit': 256})
    with open(os.path.join(config_folder, 'vc_assignments.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=1,
             segmentation_path=seg_path, vc_volume_path=vc_vol_path,
             med_expression_path=med_expression_path, output_path=output_path)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Computing gene expressions failed")


def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)


def create_auxiliary_gene_file(meds_path, out_file, return_result=False):
    all_genes_dset = 'genes'
    names_dset = 'gene_names'
    dset = 't00000/s00/0/cells'

    # get all the med files in the image folder
    med_files = glob(os.path.join(meds_path, "*MED.h5"))
    gene_names = [os.path.splitext(os.path.basename(f))[0][:-4] for f in med_files]

    # find out where the gene name actually starts (the 5th dash-separated word)
    name_start = find_nth(gene_names[0], '-', 4) + 1

    # cut all the preceeding prospr-... part
    gene_names = [name[name_start:] for name in gene_names]
    num_genes = len(med_files)
    assert len(gene_names) == num_genes, "%i, %i" % (len(gene_names), len(num_genes))

    with h5py.File(med_files[0], 'r') as f:
        spatial_shape = f[dset].shape

    shape = (num_genes,) + spatial_shape

    # iterate through med files and write down binarized into one file
    with h5py.File(out_file) as f:
        out_dset = f.create_dataset(all_genes_dset, shape=shape, dtype='bool',
                                    chunks=(1, 64, 64, 64), compression='gzip')

        for i, file_name in enumerate(tqdm(med_files)):
            with h5py.File(file_name, 'r') as f2:
                ds = f2[dset]
                this_shape = ds.shape
                if this_shape != spatial_shape:
                    raise RuntimeError("Incompatible shapes %s, %s" % (str(this_shape), str(spatial_shape)))
                data = f2[dset][:]
            out_dset[i] = data

        gene_names_ascii = [n.encode('ascii', 'ignore') for n in gene_names]
        f.create_dataset(names_dset, data=gene_names_ascii, dtype='S40')

    if return_result:
        # reload the binarized version
        with h5py.File(out_file, 'r') as f:
            all_genes = f[all_genes_dset][:]
        return all_genes
