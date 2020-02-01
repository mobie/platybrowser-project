import unittest
import numpy as np
import vigra
import h5py


class TestExport(unittest.TestCase):
    # test that we don't have any disconnected components
    def test_postprocess(self):
        scale = 3
        path = '../../data/0.3.1/segmentations/sbem-6dpf-1-whole-segmented-cells-labels.h5'
        with h5py.File(path, 'r') as f:
            ds = f['t00000/s00/%i/cells' % scale]
            seg = ds[:].astype('uint32')

        print(seg.shape)
        n_ids1 = len(np.unique(seg))
        seg = vigra.analysis.labelVolumeWithBackground(seg)
        n_ids2 = len(np.unique(seg))
        self.assertEqual(n_ids1, n_ids2)


if __name__ == '__main__':
    unittest.main()
