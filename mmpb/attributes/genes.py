import os
import json
from glob import glob

import luigi
import numpy as np
from tqdm import tqdm
from elf.io import open_file
from pybdv.util import get_key
from pybdv.metadata import get_data_path

from ..extension.attributes import GenesLocal, GenesSlurm
from ..extension.attributes import VCAssignmentsLocal, VCAssignmentsSlurm


def gene_assignment_table(segm_file, genes_file, table_file, labels,
                          tmp_folder, target, n_threads=8):
    task = GenesSlurm if target == 'slurm' else GenesLocal
    if os.path.splitext(segm_file) == '.n5':
        seg_dset = 'setup0/timepoint0/s4'
    else:
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


def vc_assignment_table(seg_path, seg_key, vc_vol_path, vc_vol_key,
                        vc_expression_path, med_expression_path, output_path,
                        tmp_folder, target, n_threads=8):
    task = VCAssignmentsSlurm if target == 'slurm' else VCAssignmentsLocal

    config_folder = os.path.join(tmp_folder, 'configs')
    config = task.default_task_config()
    # this is very ram hungry because we load all the genes at once
    config.update({'threads_per_job': n_threads, 'mem_limit': 256})
    with open(os.path.join(config_folder, 'vc_assignments.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=1,
             segmentation_path=seg_path, segmentation_key=seg_key,
             vc_volume_path=vc_vol_path, vc_volume_key=vc_vol_key,
             vc_expression_path=vc_expression_path,
             med_expression_path=med_expression_path, output_path=output_path)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Computing gene expressions failed")


def create_auxiliary_gene_file(meds_root, out_file, return_result=False):
    all_genes_dset = 'genes'
    names_dset = 'gene_names'

    # get all the prospr gene xmls in the image folder
    med_files = glob(os.path.join(meds_root, "prospr*.xml"))
    # filter out prospr files that are not genes (=semgneted regions and virtual cells)
    med_files = [name for name in med_files if 'segmented' not in name]
    med_files = [name for name in med_files if 'virtual' not in name]

    # get the gene names from filenames
    gene_names = [os.path.splitext(os.path.basename(f))[0] for f in med_files]
    # cut all the preceeding prospr-... part
    gene_names = ['-'.join(name.split('-')[4:]) for name in gene_names]
    num_genes = len(gene_names)
    assert num_genes == len(med_files)

    # get the data paths from the xmls
    med_files = [get_data_path(med_file, return_absolute_path=True)
                 for med_file in med_files]

    is_h5 = os.path.splitext(med_files[0])[1] == '.h5'
    med_key = get_key(is_h5, time_point=0, setup_id=0, scale=0)
    with open_file(med_files[0], 'r') as f:
        spatial_shape = f[med_key].shape

    shape = (num_genes,) + spatial_shape

    # iterate through med files and write down binarized into one file
    with open_file(out_file) as f:
        out_dset = f.create_dataset(all_genes_dset, shape=shape, dtype='bool',
                                    chunks=(1, 64, 64, 64), compression='gzip')
        out_dset.n_threads = 8

        for i, med_file in enumerate(tqdm(med_files)):
            is_h5 = os.path.splitext(med_file)[1] == '.h5'
            med_key = get_key(is_h5, time_point=0, setup_id=0, scale=0)
            with open_file(med_file, 'r') as f2:
                ds = f2[med_key]
                this_shape = ds.shape
                if this_shape != spatial_shape:
                    raise RuntimeError("Incompatible shapes %s, %s" % (str(this_shape),
                                                                       str(spatial_shape)))
                ds.n_threads = 8
                data = ds[:]
            out_dset[i] = data

        gene_names_ascii = [n.encode('ascii', 'ignore') for n in gene_names]
        f.create_dataset(names_dset, data=gene_names_ascii, dtype='S40')

    if return_result:
        # reload the binarized version
        with open_file(out_file, 'r') as f:
            all_genes = f[all_genes_dset][:]
        return all_genes
