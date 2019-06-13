import unittest
import os
import json
import sys
import numpy as np
from shutil import rmtree
sys.path.append('../..')


# check new version of gene mapping against original
class TestGeneAttributes(unittest.TestCase):
    tmp_folder = 'tmp_genes'
    test_file = 'tmp_genes/test_table.csv'

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def load_table(self, table_file):
        table = np.genfromtxt(table_file, delimiter='\t', skip_header=1,
                              dtype='float32')
        return table

    def test_genes(self):
        from scripts.attributes.genes import write_genes_table
        from scripts.extension.attributes import GenesLocal
        from scripts.files import get_h5_path_from_xml

        # load original genes table
        original_table_file = '../../data/0.0.0/tables/em-segmented-cells-labels/genes.csv'
        original_table = self.load_table(original_table_file)
        self.assertEqual(original_table.dtype, np.dtype('float32'))
        labels = original_table[:, 0].astype('uint64')

        # compute and load the genes table
        segm_file = '../../data/0.0.0/segmentations/em-segmented-cells-labels.h5'
        genes_file = '../../data/0.0.0/misc/meds_all_genes.xml'
        genes_file = get_h5_path_from_xml(genes_file)
        table_file = self.test_file

        # write the global config
        config_folder = os.path.join(self.tmp_folder, 'configs')
        os.makedirs(config_folder, exist_ok=True)
        conf = GenesLocal.default_global_config()
        shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        conf.update({'shebang': shebang})
        with open(os.path.join(config_folder, 'global.config'), 'w') as f:
            json.dump(conf, f)

        print("Start computation ...")
        write_genes_table(segm_file, genes_file, table_file, labels,
                          self.tmp_folder, 'local', 8)
        table = self.load_table(table_file)

        # make sure new and old table agree
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))


if __name__ == '__main__':
    unittest.main()
