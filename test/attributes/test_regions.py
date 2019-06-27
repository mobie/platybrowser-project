import unittest
import sys
import os
import json
from shutil import rmtree

import numpy as np
import pandas
import h5py
from cluster_tools.node_labels import NodeLabelWorkflow
sys.path.append('../..')


# check new version of gene mapping against original
class TestRegions(unittest.TestCase):
    tmp_folder = 'tmp'

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def check_result(self, table, base_table,
                     seg_path, seg_key,
                     tissue_path, tissue_key):

        resolution = [.1, .08, .08]

        with h5py.File(seg_path, 'r') as fseg, h5py.File(tissue_path, 'r') as ftis:
            ds_seg = fseg[seg_key]
            ds_tis = ftis[tissue_key]
            assert ds_seg.shape == ds_tis.shape, "%s, %s" % (str(ds_seg.shape),
                                                             str(ds_tis.shape))

            names = ftis['semantic_names'][:]
            mapped_ids = ftis['semantic_mapping'][:]

            for name, tis_ids in zip(names, mapped_ids):

                # these are too big
                if name in ('cuticle', 'neuropil'):
                    continue
                print("Checking", name)

                col = table[name].values
                seg_ids = np.where(col == 1.)[0]
                for seg_id in seg_ids:
                    row = base_table.iloc[seg_id]
                    bb_start = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
                    bb_stop = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
                    bb = tuple(slice(int(mi / re), int(ma / re))
                               for mi, ma, re in zip(bb_start, bb_stop, resolution))

                    seg = ds_seg[bb]
                    tis = ds_tis[bb]

                    seg_mask = seg == seg_id
                    self.assertGreater(np.in1d(tis[seg_mask], tis_ids).sum(), 0)

    def test_regions(self):
        from scripts.attributes.region_attributes import region_attributes

        image_folder = '../../data/0.0.0/images'
        segmentation_folder = '../../data/0.0.0/segmentations'
        seg_path = os.path.join(segmentation_folder,
                                'sbem-6dpf-1-whole-segmented-cells-labels.h5')
        seg_key = 't00000/s00/0/cells'
        with h5py.File(seg_path, 'r') as f:
            max_id = f[seg_key].attrs['maxId']

        output_path = os.path.join(self.tmp_folder, 'table-test.csv')
        label_ids = np.arange(max_id + 1, dtype='uint64')

        # write the global config
        config_folder = os.path.join(self.tmp_folder, 'configs')
        os.makedirs(config_folder, exist_ok=True)
        conf = NodeLabelWorkflow.get_config()['global']
        shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        conf.update({'shebang': shebang})
        with open(os.path.join(config_folder, 'global.config'), 'w') as f:
            json.dump(conf, f)

        target = 'local'
        max_jobs = 8
        region_attributes(seg_path, output_path,
                          image_folder, segmentation_folder,
                          label_ids, self.tmp_folder, target, max_jobs)

        table = pandas.read_csv(output_path, sep='\t')
        assert len(table) == max_id + 1

        base_path = '../../data/0.0.0/tables/sbem-6dpf-1-whole-segmented-cells-labels/default.csv'
        base_table = pandas.read_csv(base_path, sep='\t')

        seg_key = 't00000/s00/2/cells'
        tissue_path = '../../data/rawdata/sbem-6dpf-1-whole-segmented-tissue-labels.h5'
        tissue_key = 't00000/s00/0/cells'
        self.check_result(table, base_table,
                          seg_path, seg_key,
                          tissue_path, tissue_key)


# this indeed fails for the glands, whereas the unittest above passes
def check_external():
    t = TestRegions()

    base_table_path = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/tables/em-segmented-cells-20190520-labels.csv'
    base_table = pandas.read_csv(base_table_path, sep='\t')

    table_path = '/g/kreshuk/pape/Work/my_projects/py_platy_browser/experiments/segmentation/tissue/regions.csv'
    table = pandas.read_csv(table_path, sep='\t')

    seg_path = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-segmented-cells-20190520-labels.h5'
    tissue_path = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-segmented-tissue-labels.h5'
    seg_key = 't00000/s00/2/cells'
    tissue_key = 't00000/s00/0/cells'

    t.check_result(table, base_table,
                   seg_path, seg_key,
                   tissue_path, tissue_key)


if __name__ == '__main__':
    # check_external()
    unittest.main()
