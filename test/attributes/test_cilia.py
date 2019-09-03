import unittest
import pandas as pd


# check the basic / default attributes
class TestCilaAttributes(unittest.TestCase):
    def test_cilia_attributes(self):
        from scripts.attributes.cilia_attributes import measure_cilia_attributes

        input_path = '../../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
        input_key = 't00000/s00/0/cells'
        resolution = [0.025, 0.01, 0.01]
        base_path = '../../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
        base = pd.read_csv(base_path, sep='\t')

        out, _ = measure_cilia_attributes(input_path, input_key, base, resolution)
        self.assertEqual(len(out), len(base))


if __name__ == '__main__':
    unittest.main()
