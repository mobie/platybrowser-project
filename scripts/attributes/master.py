import os
import h5py

from .base_attributes import base_attributes
from .map_objects import map_objects
from .genes import write_genes_table
from .morphology import write_morphology_cells, write_morphology_nuclei
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
                                correct_anchors=True)

    # make table with mapping to other objects
    # nuclei, cellular models (TODO), ...
    map_out = os.path.join(table_folder, 'objects.csv')
    map_paths = [get_seg_path(folder, 'em-segmented-nuclei-labels', seg_key)]
    map_keys = [seg_key]
    map_names = ['nucleus_id']
    map_objects(label_ids, seg_path, seg_key, map_out,
                map_paths, map_keys, map_names,
                tmp_folder, target, max_jobs)

    # make table with gene mapping
    # TODO we need to make sure that this file is always copied / updated
    # before this is called !
    aux_gene_xml = os.path.join(folder, 'misc', 'meds_all_genes.xml')
    aux_gene_path = get_h5_path_from_xml(aux_gene_xml)
    if not os.path.exists(aux_gene_path):
        raise RuntimeError("Can't find auxiliary gene file")
    gene_out = os.path.join(table_folder, 'genes.csv')
    write_genes_table(seg_path, aux_gene_path, gene_out, label_ids,
                      tmp_folder, target)

    # make table with morphology
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    write_morphology_cells(seg_path, base_out, map_out, morpho_out)

    # TODO additional tables:
    # regions / semantics
    # ???


def make_nucleus_tables(folder, name, tmp_folder, resolution,
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

    # make the morphology attribute table
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    write_morphology_nuclei(seg_path, base_out, morpho_out)

    # TODO additional tables:
    # ???
