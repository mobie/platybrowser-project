import unittest
import pandas as pd
import numpy as np


# check the basic / default attributes
class TestCilaAttributes(unittest.TestCase):

    def check_attributes(self, base, attributes, resolution):
        base.set_index('label_id')
        for cid, attrs in enumerate(attributes):

            clen = attrs[0]
            if clen == 0:
                continue

            row = base.loc[cid]
            bb_min = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
            bb_max = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
            bb = [int((ma - mi) / res) for mi, ma, res in zip(bb_min, bb_max, resolution)]
            diag_len = np.sqrt(sum([b*b for b in bb]))
            self.assertGreater(clen, diag_len)

    def test_cilia_attributes(self):
        from scripts.attributes.cilia_attributes import measure_cilia_attributes

        input_path = '../../data/0.5.1/segmentations/sbem-6dpf-1-whole-segmented-cilia-labels.h5'
        input_key = 't00000/s00/0/cells'
        resolution = [0.025, 0.01, 0.01]
        base_path = '../../data/0.5.1/tables/sbem-6dpf-1-whole-segmented-cilia-labels/default.csv'
        base = pd.read_csv(base_path, sep='\t')

        out, _ = measure_cilia_attributes(input_path, input_key, base, resolution)
        # out = np.load('out.npy')
        self.assertEqual(len(out), len(base))
        self.check_attributes(base, out, resolution)
        # np.save('out.npy', out)

if __name__ == '__main__':
    unittest.main()
