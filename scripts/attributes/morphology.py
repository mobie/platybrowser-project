import luigi
import os
import json
from ..extension.attributes import MorphologyWorkflow


def write_config(config_folder, config):
    # something eats a lot of threads, let's at least reserve a few ...
    config.update({'mem_limit': 24, 'threads_per_job': 4})
    with open(os.path.join(config_folder, 'morphology.config'), 'w') as f:
        json.dump(config, f)


def write_morphology_nuclei(seg_path, raw_path, table_in_path, table_out_path,
                            n_labels, resolution, tmp_folder, target, max_jobs):
    """
    Write csv files of morphology stats for the nucleus segmentation

    seg_path - string, file path to nucleus segmentation
    raw_path - string, file path to raw data
    cell_table_path - string, file path to cell table
    table_in_path - string, file path to nucleus table
    table_out_path - string, file path to save new nucleus table
    n_labels - int, number of labels
    resolution - list or tuple, resolution of segmentation at scale 0
    tmp_folder - string, temporary folder
    target - string, computation target (slurm or local)
    max_jobs - maximal number of jobs
    """
    task = MorphologyWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')
    write_config(config_folder, task.get_config()['morphology'])

    seg_scale = 0
    min_size = 18313  # Kimberly's lower size cutoff for nuclei
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder, target=target,
             segmentation_path=seg_path, in_table_path=table_in_path,
             output_path=table_out_path, resolution=list(resolution),
             seg_scale=seg_scale, raw_scale=3, min_size=min_size, max_size=None,
             raw_path=raw_path, prefix='nuclei', number_of_labels=n_labels)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Nucleus morphology computation failed")


def write_morphology_cells(seg_path, table_in_path, cell_nuc_mapping_path, table_out_path,
                           n_labels, resolution, tmp_folder, target, max_jobs):

    """
    Write csv files of morphology stats for both the nucleus and cell segmentation

    seg_path - string, file path to cell segmentation
    table_in_path - string, file path to cell table
    cell_nuc_mapping_path - string, file path to numpy array mapping cells to nuclei
        (first column cell id, second nucleus id)
    table_out_path - string, file path to save new cell table
    n_labels - int, number of labels
    resolution - list or tuple, resolution of segmentation at scale 0
    tmp_folder - string, temporary folder
    target - string, computation target (slurm or local)
    max_jobs - maximal number of jobs
    """
    task = MorphologyWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')
    write_config(config_folder, task.get_config()['morphology'])

    seg_scale = 2
    min_size = 88741  # Kimberly's lower size cutoff for cells
    max_size = 600000000  # Kimberly's upper size cutoff for cells
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder, target=target,
             segmentation_path=seg_path, in_table_path=table_in_path,
             output_path=table_out_path, resolution=list(resolution),
             seg_scale=seg_scale, min_size=min_size, max_size=max_size,
             mapping_path=cell_nuc_mapping_path, prefix='cells',
             number_of_labels=n_labels)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Cell morphology computation failed")
