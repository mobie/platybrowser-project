import luigi
import os
import json
from ..extension.attributes import MorphologyWorkflow


def write_config(config_folder, config):
    config.update({'mem_limit': 24})
    with open(os.path.join(config_folder, 'morphology.config'), 'w') as f:
        json.dump(config, f)


def write_morphology_nuclei(raw_path, nucleus_seg_path, chromatin_seg_path,
                            table_in_path, table_out_path,
                            tmp_folder, target, max_jobs):
    """
    Write csv files of morphology stats for the nucleus segmentation

    raw_path - string, file path to raw data
    nucleus_seg_path - string, file path to nucleus segmentation
    chromatin_seg_path - string, file path to chromatin segmentation
    table_in_path - string, file path to nucleus table
    table_out_path - string, file path to save new nucleus table
    tmp_folder - string, temporary folder
    target - string, computation target (slurm or local)
    max_jobs - maximal number of jobs
    """
    task = MorphologyWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')
    write_config(config_folder, task.get_config()['morphology'])

    scale = 3  # this is the scale of the raw data
    min_size = 18313
    max_bb = None
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder, target=target,
             compute_cell_features=False, raw_path=raw_path,
             nucleus_segmentation_path=nucleus_seg_path,
             chromatin_segmentation_path=chromatin_seg_path,
             in_table_path=table_in_path,
             output_path=table_out_path,
             scale=scale, min_size=min_size, max_bb=max_bb)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Nucleus morphology computation failed")


def write_morphology_cells(raw_path, cell_seg_path, nucleus_seg_path,
                           table_in_path, table_out_path,
                           nucleus_mapping_path, region_path,
                           resolution, tmp_folder, target, max_jobs):

    """
    Write csv files of morphology stats for both the nucleus and cell segmentation

    cell_seg_path - string, file path to cell segmentation
    table_in_path - string, file path to cell table
    nucleus_mapping_path - string, file path to numpy array mapping cells to nuclei
        (first column cell id, second nucleus id)
    table_out_path - string, file path to save new cell table
    tmp_folder - string, temporary folder
    target - string, computation target (slurm or local)
    max_jobs - maximal number of jobs
    """
    task = MorphologyWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')
    write_config(config_folder, task.get_config()['morphology'])

    scale = 3  # this is the scale of the raw data
    min_size = 88741  # Kimberly's lower size cutoff for cells
    max_size = 600000000  # Kimberly's upper size cutoff for cells
    max_bb = 454000
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder, target=target,
             compute_cell_features=True, raw_path=raw_path,
             cell_segmentation_path=cell_seg_path,
             nucleus_segmentation_path=nucleus_seg_path,
             in_table_path=table_in_path,
             output_path=table_out_path,
             scale=scale, max_bb=max_bb,
             min_size=min_size, max_size=max_size,
             nucleus_mapping_path=nucleus_mapping_path,
             region_mapping_path=region_path)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Cell morphology computation failed")
