import os
import h5py

from .base_attributes import base_attributes
from .cell_nucleus_mapping import map_cells_to_nuclei
from .genes import write_genes_table
from .morphology import write_morphology_cells, write_morphology_nuclei
from .region_attributes import region_attributes
from ..files import get_h5_path_from_xml


def get_seg_path(folder, name, key):
    xml_path = os.path.join(folder, 'segmentations', '%s.xml' % name)
    path = get_h5_path_from_xml(xml_path, return_absolute_path=True)
    assert os.path.exists(path), path
    with h5py.File(path, 'r') as f:
        assert key in f, "%s not in %s" % (key, str(list(f.keys())))
    return path


def make_cell_tables(folder, name, tmp_folder, resolution,
                     target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = 't00000/s00/0/cells'
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    label_ids = base_attributes(seg_path, seg_key, base_out, resolution,
                                tmp_folder, target=target, max_jobs=max_jobs,
                                correct_anchors=False)

    # make table with cell nucleus mapping
    nuc_mapping_table = os.path.join(table_folder, 'cells_to_nuclei.csv')
    nuc_path = get_seg_path(folder, 'sbem-6dpf-1-whole-segmented-nuclei-labels', seg_key)
    map_cells_to_nuclei(label_ids, seg_path, nuc_path, nuc_mapping_table,
                        tmp_folder, target, max_jobs)

    # make table with gene mapping
    aux_gene_xml = os.path.join(folder, 'misc', 'prospr-6dpf-1-whole_meds_all_genes.xml')
    aux_gene_path = get_h5_path_from_xml(aux_gene_xml, return_absolute_path=True)
    if not os.path.exists(aux_gene_path):
        raise RuntimeError("Can't find auxiliary gene file @ %s" % aux_gene_path)
    gene_out = os.path.join(table_folder, 'genes.csv')
    write_genes_table(seg_path, aux_gene_path, gene_out, label_ids,
                      tmp_folder, target)

    # make table with morphology
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    n_labels = len(label_ids)
    write_morphology_cells(seg_path, base_out, nuc_mapping_table, morpho_out,
                           n_labels, resolution, tmp_folder,
                           target, max_jobs)

    # region and semantic mapping
    region_out = os.path.join(table_folder, 'regions.csv')
    # need to make sure the inputs are copied / updated in
    # the segmentation folder beforehand
    image_folder = os.path.join(folder, 'images')
    segmentation_folder = os.path.join(folder, 'segmentations')
    region_attributes(seg_path, region_out,
                      image_folder, segmentation_folder,
                      label_ids, tmp_folder, target, max_jobs)


def make_nucleus_tables(folder, name, tmp_folder, resolution,
                        target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = 't00000/s00/0/cells'
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    label_ids = base_attributes(seg_path, seg_key, base_out, resolution,
                                tmp_folder, target=target, max_jobs=max_jobs,
                                correct_anchors=True)

    xml_raw = os.path.join(folder, 'images', 'sbem-6dpf-1-whole-raw.xml')
    raw_path = get_h5_path_from_xml(xml_raw, return_absolute_path=True)
    # make the morphology attribute table
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    n_labels = len(label_ids)
    write_morphology_nuclei(seg_path, raw_path, base_out, morpho_out,
                            n_labels, resolution, tmp_folder,
                            target, max_jobs)

    # TODO additional tables:
    # ???


def make_cilia_tables(folder, name, tmp_folder, resolution,
                      target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = 't00000/s00/0/cells'
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    base_attributes(seg_path, seg_key, base_out, resolution,
                    tmp_folder, target=target, max_jobs=max_jobs,
                    correct_anchors=True)
    # TODO additional tables:
    # ???
