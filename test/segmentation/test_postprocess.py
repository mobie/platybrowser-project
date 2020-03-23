import os
import unittest
from shutil import rmtree

import numpy as np
import nifty.ground_truth as ngt
import nifty.tools as nt
import vigra
import z5py
from paintera_tools import (convert_to_paintera_format,
                            downscale,
                            postprocess)
from mmpb.export.check_segmentation import check_connected_components


def max_overlaps(a, b):
    ovlp = ngt.overlap(a, b)
    max_ols = [ovlp.overlapArrays(seg_id, sorted=True) for seg_id in range(int(a.max()) + 1)]
    max_ols = [max_ol[0][0] if len(max_ol[0]) > 0 else 0 for max_ol in max_ols]
    return np.array(max_ols, dtype='uint64')


class TestPostprocess(unittest.TestCase):
    tmp_folder = './tmp'
    test_data_path = './test_data.n5'

    @staticmethod
    def make_test_data():
        in_path = '/g/kreshuk/data/cremi/example/sampleA.n5'
        raw_key = 'volumes/raw/s0'
        ws_key = 'volumes/segmentation/watershed'
        seg_key = 'volumes/segmentation/multicut'
        bd_key = 'volumes/boundaries'

        with z5py.File(in_path, 'r') as f:
            ds = f[ws_key]
            ds.n_threasds = 8

            halo = [25, 512, 512]
            bb = tuple(slice(sh // 2 - ha, sh // 2 + ha)
                       for sh, ha in zip(ds.shape, halo))

            ws = ds[bb]

            ds = f[seg_key]
            ds.n_threasds = 8
            seg = ds[bb]

            ds = f[bd_key]
            ds.n_threads = 8
            bd = ds[bb]

            ds = f[raw_key]
            ds.n_threads = 8
            raw = ds[bb]

            chunks = ds.chunks

        print("Run ccs ...")
        ws = vigra.analysis.labelVolumeWithBackground(ws.astype('uint32')).astype('uint64')
        seg = vigra.analysis.labelVolumeWithBackground(seg.astype('uint32')).astype('uint64')
        print("ccs done")

        node_labels = max_overlaps(ws, seg)
        unique_labels = np.unique(node_labels)

        n_merge = 50
        merge_to = 3
        has_merge = []

        node_labels_merged = node_labels.copy()
        for ii in range(n_merge):
            merge_id = np.random.choice(unique_labels)
            while merge_id in has_merge:
                merge_id = np.random.choice(unique_labels)

            for jj in range(merge_to):
                merge_to_id = np.random.choice(unique_labels)
                while merge_to_id in has_merge:
                    merge_to_id = np.random.choice(unique_labels)
                node_labels_merged[node_labels_merged == merge_to_id] = merge_id

            unique_labels = np.unique(node_labels_merged)
            has_merge.append(merge_id)

        seg_merged = nt.take(node_labels_merged, ws)
        assert seg_merged.shape == seg.shape

        print("Write outputs")
        out_path = './test_data.n5'
        out_raw_key = 'volumes/raw/s0'
        with z5py.File(out_path, 'a') as f:
            ds = f.create_dataset('volumes/seg/ref', data=seg, chunks=chunks, compression='gzip')
            ds.attrs['maxId'] = int(seg.max())

            ds = f.create_dataset('volumes/seg/merged', data=seg_merged, chunks=chunks, compression='gzip')
            ds.attrs['maxId'] = int(seg_merged.max())

            ds = f.create_dataset('volumes/ws', data=ws, chunks=chunks, compression='gzip')
            ds.attrs['maxId'] = int(ws.max())

            f.create_dataset('node_labels/ref', data=node_labels, compression='gzip')
            f.create_dataset('node_labels/merged', data=node_labels_merged, compression='gzip')
            f.create_dataset('volumes/boundaries', data=bd, compression='gzip', chunks=chunks)
            f.create_dataset(out_raw_key, data=raw, compression='gzip', chunks=chunks)

        print("Make paintera dataset")
        # make the paintera dataset
        tmp_paintera = './tmp_paintera'
        scale_factors = [[1, 2, 2], [1, 2, 2]]
        halos = [[1, 2, 2], [1, 2, 2]]
        target = 'local'
        max_jobs = 16
        downscale(out_path, out_raw_key, 'volumes/raw', scale_factors, halos,
                  tmp_paintera, target, max_jobs)

        convert_to_paintera_format(out_path, 'volumes/raw', 'volumes/ws', 'volumes/paintera',
                                   label_scale=0, resolution=[1, 1, 1],
                                   tmp_folder=tmp_paintera, target=target,
                                   max_jobs=max_jobs, max_threads=max_jobs,
                                   assignment_path=out_path, assignment_key='node_labels/merged',
                                   convert_to_label_multisets=True, restrict_sets=[-1, -1])

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _test_connected_components(self, seg_path, seg_key, tmp_folder=None):
        path = self.test_data_path
        target = 'local'
        max_jobs = 16

        ws_key = 'volumes/ws'
        tmp_folder = self.tmp_folder if tmp_folder is None else tmp_folder
        passed = check_connected_components(path, ws_key, seg_path, seg_key,
                                            tmp_folder, target, max_jobs,
                                            margin=1)
        return passed

    def test_connected_components_true(self):
        passed = self._test_connected_components(self.test_data_path, 'volumes/seg/ref')
        self.assertTrue(passed)

    def test_connected_components_false(self):
        passed = self._test_connected_components(self.test_data_path, 'volumes/seg/merged')
        self.assertFalse(passed)

    def _postprocess(self, label_segmentation=False, size_threshold=None, target_number=None):
        path = self.test_data_path
        boundary_key = 'volumes/boundaries'
        paintera_key = 'volumes/paintera'

        out_path = os.path.join(self.tmp_folder, 'data.n5')
        out_key = 'seg/pp'

        target = 'local'
        max_jobs = 16

        tmp_postprocess = self.tmp_folder
        postprocess(path, paintera_key,
                    path, boundary_key,
                    tmp_folder=tmp_postprocess,
                    target=target, max_jobs=max_jobs,
                    n_threads=max_jobs, size_threshold=size_threshold,
                    target_number=target_number,
                    label=label_segmentation,
                    output_path=out_path, output_key=out_key)
        return out_path, out_key

    def check_volume(self, seg_path, seg_key, target_number=None):
        with z5py.File(seg_path, 'r') as f:
            seg = f[seg_key][:].astype('uint32')

        nodes = np.unique(seg)
        seg_cc = vigra.analysis.labelVolumeWithBackground(seg)
        nodes_cc = np.unique(seg_cc)
        self.assertEqual(len(nodes), len(nodes_cc))
        if target_number is not None:
            self.assertEqual(len(nodes_cc), target_number)

    def test_postprocess_ccs(self):
        seg_path, seg_key = self._postprocess(label_segmentation=True)
        passed = self._test_connected_components(seg_path, seg_key,
                                                 os.path.join(self.tmp_folder, 'tmp2'))
        self.assertTrue(passed)
        self.check_volume(seg_path, seg_key)

    def test_postprocess_ccs_and_size_thresh(self):
        target_number = 1200
        seg_path, seg_key = self._postprocess(label_segmentation=True, target_number=target_number)
        passed = self._test_connected_components(seg_path, seg_key,
                                                 os.path.join(self.tmp_folder, 'tmp2'))
        self.assertTrue(passed)
        self.check_volume(seg_path, seg_key, target_number)


if __name__ == '__main__':
    # TestPostprocess.make_test_data()
    unittest.main()
