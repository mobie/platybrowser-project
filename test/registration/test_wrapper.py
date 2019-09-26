import os
import json
import unittest

import luigi
import numpy as np
import imageio
from shutil import rmtree
from scripts.extension.registration import ApplyRegistrationLocal


class TestRegistrationWrapper(unittest.TestCase):
    tmp_folder = './tmp_regestration'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _apply_registration(self, in_path, trafo_file, interpolation, file_format, dtype):
        out_path = os.path.join(self.tmp_folder, 'out')

        in_list = [in_path]
        out_list = [out_path]
        in_file = os.path.join(self.tmp_folder, 'in_list.json')
        with open(in_file, 'w') as f:
            json.dump(in_list, f)
        out_file = os.path.join(self.tmp_folder, 'out_list.json')
        with open(out_file, 'w') as f:
            json.dump(out_list, f)

        task = ApplyRegistrationLocal
        conf_dir = os.path.join(self.tmp_folder, 'configs')
        os.makedirs(conf_dir, exist_ok=True)

        global_conf = task.default_global_config()
        shebang = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs',
                               'platybrowser/bin/python')
        global_conf.update({'shebang': shebang})
        with open(os.path.join(conf_dir, 'global.config'), 'w') as f:
            json.dump(global_conf, f)

        task_conf = task.default_task_config()
        task_conf.update({'threads_per_job': 8, 'ResultImagePixelType': dtype})
        with open(os.path.join(conf_dir, 'apply_registration.config'), 'w') as f:
            json.dump(task_conf, f)

        t = task(tmp_folder=self.tmp_folder, config_dir=conf_dir, max_jobs=1,
                 input_path_file=in_file, output_path_file=out_file, transformation_file=trafo_file,
                 interpolation=interpolation, output_format=file_format)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        return out_path

    # only makes sense for nearest neighbor interpolation
    def check_result(self, in_path, res_path, check_range=False):
        res = imageio.volread(res_path)
        exp = imageio.volread(in_path).astype(res.dtype)

        if check_range:
            min_res = res.min()
            min_exp = exp.min()
            self.assertEqual(min_res, min_exp)
            max_res = res.max()
            max_exp = exp.max()
            self.assertEqual(max_res, max_exp)
        else:
            un_res = np.unique(res)
            un_exp = np.unique(exp)
            self.assertTrue(np.array_equal(un_exp, un_res))

    def test_nearest_mask(self):
        trafo_dir = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/0.0.0/transformations'
        # This is the full transformation, but it takes a lot of time!
        # trafo_file = os.path.join(trafo_dir, 'TransformParameters.BSpline10-3Channels.0.txt')
        # For now, we use the similarity trafo to save time
        trafo_file = os.path.join(trafo_dir, 'TransformParameters.Similarity-3Channels.0.txt')

        in_path = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/9.9.9/ProSPr/stomach.tif'
        out_path = self._apply_registration(in_path, trafo_file, 'nearest', 'tif', 'unsigned char')

        out_path = out_path + '-ch0.tif'
        self.assertTrue(os.path.exists(out_path))
        self.check_result(in_path, out_path)

    def test_nearest_seg(self):
        trafo_dir = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/0.0.0/transformations'
        # This is the full transformation, but it takes a lot of time!
        # trafo_file = os.path.join(trafo_dir, 'TransformParameters.BSpline10-3Channels.0.txt')
        # For now, we use the similarity trafo to save time
        trafo_file = os.path.join(trafo_dir, 'TransformParameters.Similarity-3Channels.0.txt')

        in_path = '/g/kreshuk/zinchenk/cell_match/data/genes/vc_volume_prospr_space_all_vc.tif'
        out_path = self._apply_registration(in_path, trafo_file, 'nearest', 'tif', 'unsigned short')

        out_path = out_path + '-ch0.tif'
        self.assertTrue(os.path.exists(out_path))
        # we can only check the range for segmentations, because individual ids might be lost
        self.check_result(in_path, out_path, check_range=True)


if __name__ == '__main__':
    unittest.main()
