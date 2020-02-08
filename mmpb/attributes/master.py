import os
from elf.io import open_file
from pybdv.metadata import get_data_path, get_bdv_format
from pybdv.util import get_key

from .base_attributes import (add_cell_criterion_column, base_attributes,
                              propagate_attributes, write_additional_table_file)
from .cell_nucleus_mapping import map_cells_to_nuclei
from .genes import gene_assignment_table, vc_assignment_table
from .morphology import write_morphology_cells, write_morphology_nuclei
from .region_attributes import region_attributes, extrapolated_intensities
from .cilia_attributes import cilia_morphology


def get_seg_path(folder, name, key=None):
    xml_path = os.path.join(folder, 'images', 'local', '%s.xml' % name)
    path = get_data_path(xml_path, return_absolute_path=True)
    assert os.path.exists(path), path
    if key is not None:
        with open_file(path, 'r') as f:
            assert key in f, "%s not in %s" % (key, path)
    return path


def get_seg_key(folder, name, scale):
    xml_path = os.path.join(folder, 'images', 'local', '%s.xml' % name)
    bdv_format = get_bdv_format(xml_path)
    if bdv_format == 'bdv.hdf5':
        return get_key(True, time_point=0, setup_id=0, scale=scale)
    elif bdv_format == 'bdv.n5':
        return get_key(False, time_point=0, setup_id=0, scale=scale)
    else:
        raise RuntimeError("Invalid bdv format: %s" % bdv_format)


def make_cell_tables(old_folder, folder, name, tmp_folder, resolution,
                     target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = get_seg_key(folder, name, scale=0)
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    label_ids = base_attributes(seg_path, seg_key, base_out, resolution,
                                tmp_folder, target=target, max_jobs=max_jobs,
                                correct_anchors=False)

    # make table with cell nucleus mapping
    nuc_mapping_table = os.path.join(table_folder, 'cells_to_nuclei.csv')
    nuc_path = get_seg_path(folder, 'sbem-6dpf-1-whole-segmented-nuclei', seg_key)
    map_cells_to_nuclei(label_ids, seg_path, nuc_path, nuc_mapping_table,
                        tmp_folder, target, max_jobs)

    # add a column with (somewhat stringent) cell criterion to the default table
    add_cell_criterion_column(base_out, nuc_mapping_table)

    # TODO need to update gene names in aux h5 file and in profile_clust ... table
    # # make table with gene mapping
    # aux_gene_xml = os.path.join(folder, 'misc', 'prospr-6dpf-1-whole_meds_all_genes.xml')
    # aux_gene_path = get_data_path(aux_gene_xml, return_absolute_path=True)
    # if not os.path.exists(aux_gene_path):
    #     raise RuntimeError("Can't find auxiliary gene file @ %s" % aux_gene_path)
    # gene_out = os.path.join(table_folder, 'genes.csv')
    # gene_assignment_table(seg_path, aux_gene_path, gene_out, label_ids,
    #                       tmp_folder, target)

    # # make table with gene mapping via VCs
    # vc_name = 'prospr-6dpf-1-whole-virtual-cells'
    # vc_vol_path = get_seg_path(folder, vc_name)
    # vc_key = get_seg_key(folder, vc_name, scale=0)
    # vc_expression_path = os.path.join(folder, 'tables', vc_name, 'profile_clust_curated.csv')
    # med_expression_path = gene_out
    # vc_out = os.path.join(table_folder, 'vc_assignments.csv')
    # vc_assignment_table(seg_path, vc_vol_path, vc_key,
    #                     vc_expression_path, med_expression_path,
    #                     vc_out, tmp_folder, target)

    # region and semantic mapping
    region_out = os.path.join(table_folder, 'regions.csv')
    # need to make sure the inputs are copied / updated in
    # the segmentation folder beforehand
    image_folder = os.path.join(folder, 'images')
    segmentation_folder = os.path.join(folder, 'segmentations')
    region_attributes(seg_path, region_out,
                      image_folder, segmentation_folder,
                      label_ids, tmp_folder, target, max_jobs)

    # make table with morphology
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    write_morphology_cells(seg_path, nuc_path,
                           base_out, morpho_out,
                           nuc_mapping_table, region_out,
                           tmp_folder, target, max_jobs)

    # mapping to extrapolated intensities
    mask_name = 'sbem-6dpf-1-whole-segmented-extrapolated'
    k1 = get_seg_key(folder, name, 3)
    k2 = get_seg_key(folder, mask_name, 0)
    extrapol_mask = os.path.join(folder, 'images', 'local', '%s.xml' % mask_name)
    extrapol_mask = get_data_path(extrapol_mask, return_absolute_path=True)
    extrapol_out = os.path.join(table_folder, 'extrapolated_intensity_correction.csv')
    extrapolated_intensities(seg_path, k1, extrapol_mask, k2,
                             extrapol_out, tmp_folder, target, max_jobs)

    # update the cell id column of the cilia cell_id_mapping table
    cilia_name = 'sbem-6dpf-1-whole-segmented-cilia'
    propagate_attributes(os.path.join(folder, 'misc',
                                      'new_id_lut_sbem-6dpf-1-whole-segmented-cells.json'),
                         os.path.join(old_folder, 'tables', cilia_name, 'cell_mapping.csv'),
                         os.path.join(folder, 'tables', cilia_name, 'cell_mapping.csv'),
                         'cell_id')

    # TODO
    # update the ganglia id mapping table

    write_additional_table_file(table_folder)


def make_nuclei_tables(old_folder, folder, name, tmp_folder, resolution,
                       target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = get_seg_key(folder, name, scale=0)
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    base_attributes(seg_path, seg_key, base_out, resolution,
                    tmp_folder, target=target, max_jobs=max_jobs,
                    correct_anchors=True)

    # make the morphology attribute table
    xml_raw = os.path.join(folder, 'images', 'sbem-6dpf-1-whole-raw.xml')
    raw_path = get_data_path(xml_raw, return_absolute_path=True)
    cell_seg_path = get_seg_path(folder, 'sbem-6dpf-1-whole-segmented-cells')
    chromatin_seg_path = get_seg_path(folder, 'sbem-6dpf-1-whole-segmented-chromatin')
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    write_morphology_nuclei(raw_path, seg_path,
                            cell_seg_path, chromatin_seg_path,
                            base_out, morpho_out,
                            tmp_folder, target, max_jobs)

    # mapping to extrapolated intensities
    mask_name = 'sbem-6dpf-1-whole-segmented-extrapolated'
    k1 = get_seg_key(folder, name, 1)
    k2 = get_seg_key(folder, mask_name, 0)
    extrapol_mask = os.path.join(folder, 'segmentations', '%s.xml' % mask_name)
    extrapol_mask = get_data_path(extrapol_mask, return_absolute_path=True)
    extrapol_out = os.path.join(table_folder, 'extrapolated_intensity_correction.csv')
    extrapolated_intensities(seg_path, k1, extrapol_mask, k2,
                             extrapol_out, tmp_folder, target, max_jobs)

    write_additional_table_file(table_folder)


def make_cilia_tables(old_folder, folder, name, tmp_folder, resolution,
                      target='slurm', max_jobs=100):
    # make the table folder
    table_folder = os.path.join(folder, 'tables', name)
    os.makedirs(table_folder, exist_ok=True)

    seg_key = get_seg_key(folder, name, scale=0)
    seg_path = get_seg_path(folder, name, seg_key)

    # make the basic attributes table
    base_out = os.path.join(table_folder, 'default.csv')
    base_attributes(seg_path, seg_key, base_out, resolution,
                    tmp_folder, target=target, max_jobs=max_jobs,
                    correct_anchors=True)

    # add cilia specific attributes (length, diameter)
    morpho_out = os.path.join(table_folder, 'morphology.csv')
    cilia_morphology(seg_path, seg_key,
                     base_out, morpho_out, resolution,
                     tmp_folder, target=target, max_jobs=max_jobs)

    # update the label id column of the cell_id_mapping table
    propagate_attributes(os.path.join(folder, 'misc',
                                      'new_id_lut_sbem-6dpf-1-whole-segmented-cilia.json'),
                         os.path.join(old_folder, 'tables', name, 'cell_mapping.csv'),
                         os.path.join(table_folder, 'cell_mapping.csv'), 'label_id')

    write_additional_table_file(table_folder)
