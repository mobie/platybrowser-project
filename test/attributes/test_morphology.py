import unittest
import sys
import os
import numpy as np
sys.path.append('../..')


# check new version of gene mapping against original
class TestMorphologyAttributes(unittest.TestCase):
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

    def test_nucleus_morphology(self):
        from scripts.attributes.morphology import write_morphology_nuclei

        # compute and load the morpho table
        seg_path = '../../data/0.0.0/segmentations/em-segmented-nuclei-labels.h5'
        table_in_path = '../../data/0.0.0/tables/em-segmented-nuclei-labels/default.csv'
        table_out_path = self.test_file
        print("Start computation ...")
        write_morphology_nuclei(seg_path, table_in_path, table_out_path)
        table = self.load_table(table_out_path)

        # load original table, make sure new and old table agree
        original_table_file = '../../data/0.0.0/tables/em-segmented-nuclei-labels/morphology.csv'
        original_table = self.load_table(original_table_file)
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))

    def test_cell_morphology(self):
        from scripts.attributes.morphology import write_morphology_cells

        # compute and load the morpho table
        seg_path = '../../data/0.0.0/segmentations/em-segmented-cells-labels.h5'
        mapping_path = '../../data/0.0.0/tables/em-segmented-cells-labels/objects.csv'
        table_in_path = '../../data/0.0.0/tables/em-segmented-cells-labels/default.csv'
        table_out_path = self.test_file
        print("Start computation ...")
        write_morphology_cells(seg_path, table_in_path, mapping_path, table_out_path)
        table = self.load_table(table_out_path)

        # load original table, make sure new and old table agree
        original_table_file = '../../data/0.0.0/tables/em-segmented-cells-labels/morphology.csv'
        original_table = self.load_table(original_table_file)
        self.assertEqual(table.shape, original_table.shape)
        self.assertTrue(np.allclose(table, original_table))


if __name__ == '__main__':
    unittest.main()
