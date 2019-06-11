import unittest
import sys
import os
import numpy as np
sys.path.append('../..')


# check new version of gene mapping against original
class TestGeneAttributes(unittest.TestCase):
    test_file = 'test_table.csv'

    def tearDown(self):
        try:
            os.remove(self.test_file)
        except OSError:
            pass

    def load_table(self, table_file):
        table = np.genfromtxt(table_file, delimiter='\t', skip_header=1,
                              dtype='float32')
        return table

    def test_genes(self):
        from scripts.attributes.genes import write_genes_table
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
        print("Start computation ...")
        write_genes_table(segm_file, genes_file, table_file, labels)
        table = self.load_table(table_file)

        # make sure new and old table agree
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))


if __name__ == '__main__':
    unittest.main()
