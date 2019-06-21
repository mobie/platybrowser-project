import unittest
import sys
import os
import json
import numpy as np
from shutil import rmtree
sys.path.append('../..')


# check new version of gene mapping against original
class TestMorphologyAttributes(unittest.TestCase):
    tmp_folder = 'tmp_morpho'

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def load_table(self, table_file):
        table = np.genfromtxt(table_file, delimiter='\t', skip_header=1,
                              dtype='float32')
        return table

    def write_global_config(self, conf):
        config_folder = os.path.join(self.tmp_folder, 'configs')
        os.makedirs(config_folder, exist_ok=True)
        shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        conf.update({'shebang': shebang})
        with open(os.path.join(config_folder, 'global.config'), 'w') as f:
            json.dump(conf, f)

    def test_nucleus_morphology(self):
        from scripts.attributes.morphology import write_morphology_nuclei
        from scripts.extension.attributes import MorphologyWorkflow
        from scripts.files import get_h5_path_from_xml

        self.write_global_config(MorphologyWorkflow.get_config()['global'])

        raw_path = '../../data/0.0.0/images/sbem-6dpf-1-whole-raw.xml'
        raw_path = get_h5_path_from_xml(raw_path)

        # compute and load the morpho table
        seg_path = '../../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'
        table_in_path = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-nuclei-labels/default.csv'
        table_out_path = os.path.join(self.tmp_folder, 'table_nuclei.csv')
        res = [.1, .08, .08]
        n_labels = np.genfromtxt(table_in_path, delimiter='\t', skip_header=1).shape[0]

        print("Start computation ...")
        write_morphology_nuclei(seg_path, raw_path, table_in_path, table_out_path,
                                n_labels, res, self.tmp_folder, 'local', 32)
        table = self.load_table(table_out_path)

        # load original table, make sure new and old table agree
        original_table_file = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-nuclei-labels/morphology.csv'
        original_table = self.load_table(original_table_file)
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))

    def test_cell_morphology(self):
        from scripts.attributes.morphology import write_morphology_cells
        from scripts.extension.attributes import MorphologyWorkflow

        self.write_global_config(MorphologyWorkflow.get_config()['global'])

        # compute and load the morpho table
        seg_path = '../../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
        mapping_path = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-cells-labels/objects.csv'
        table_in_path = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-cells-labels/default.csv'
        table_out_path = os.path.join(self.tmp_folder, 'table_cells.csv')
        res = [.025, .02, .02]
        n_labels = np.genfromtxt(table_in_path, delimiter='\t', skip_header=1).shape[0]

        print("Start computation ...")
        write_morphology_cells(seg_path, table_in_path, mapping_path, table_out_path,
                               n_labels, res, self.tmp_folder, 'local', 32)
        table = self.load_table(table_out_path)

        # make sure new and old table agree
        original_table_file = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-cells-labels/morphology.csv'
        original_table = self.load_table(original_table_file)
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))


if __name__ == '__main__':
    unittest.main()
