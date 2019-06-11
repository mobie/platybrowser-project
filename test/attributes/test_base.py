import unittest
import sys
import os
import pandas
import h5py
from shutil import rmtree
sys.path.append('../..')


# check new version of gene mapping against original
class TestGeneAttributes(unittest.TestCase):
    tmp_folder = 'tmp'

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_genes(self):
        from scripts.attributes.base_attributes import base_attributes

        input_path = '../../data/0.0.0/segmentations/em-segmented-nuclei-labels.h5'
        input_key = 't00000/s00/0/cells'
        output_path = os.path.join(self.tmp_folder, 'table-test.csv')
        target = 'local'
        max_jobs = 32
        resolution = [0.1, 0.08, 0.08]
        base_attributes(input_path, input_key, output_path, resolution,
                        self.tmp_folder, target, max_jobs, correct_anchors=False)

        table = pandas.read_csv(output_path, sep='\t')
        print("Checking attributes ...")
        with h5py.File(input_path, 'r') as f:
            ds = f[input_key]
            for row in table.itertuples(index=False):
                label_id = int(row.label_id)
                if label_id == 0:
                    continue

                # NOTE we increase the bounding box by 1 due to potential rounding artifacts
                # check bounding box
                bb_min = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
                bb_max = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
                bb = tuple(slice(int(min_ / res) - 1, int(max_ / res) + 2)
                           for min_, max_, res in zip(bb_min, bb_max, resolution))
                seg = ds[bb]

                # TODO check anchor once we have anchor correction
                # anchor = [row.anchor_z, row.anchor_y, row.anchor_x]
                # anchor = tuple(anch // res - b.start for anch, res, b in zip(anchor, resolution, bb))
                # anchor_id = seg[anchor]
                # self.assertEqual(anchor_id, label_id)

                label_mask = seg == label_id
                n_pixels = label_mask.sum()
                self.assertGreater(n_pixels, 0)

                # check the pixel size
                pixel_size = int(row.n_pixels)
                self.assertEqual(pixel_size, n_pixels)


if __name__ == '__main__':
    unittest.main()
