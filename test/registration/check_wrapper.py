import os
import json

import luigi
from scripts.extension.registration import ApplyRegistrationLocal


def check_wrapper():
    in_path = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/9.9.9/ProSPr/stomach.tif'

    tmp_folder = os.path.abspath('tmp_registration')
    out_path = os.path.join(tmp_folder, 'stomach_prospr_registered')

    in_list = [in_path]
    out_list = [out_path]
    in_file = './in_list.json'
    with open(in_file, 'w') as f:
        json.dump(in_list, f)
    out_file = './out_list.json'
    with open(out_file, 'w') as f:
        json.dump(out_list, f)

    task = ApplyRegistrationLocal
    conf_dir = './configs'
    os.makedirs(conf_dir, exist_ok=True)

    global_conf = task.default_global_config()
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_conf.update({'shebang': shebang})
    with open(os.path.join(conf_dir, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    trafo_dir = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/0.0.0/transformations'

    # This is the full transformation, but it takes a lot of time!
    trafo = os.path.join(trafo_dir, 'TransformParameters.BSpline10-3Channels.0.txt')

    # For now, we use the similarity trafo to save time
    trafo = os.path.join(trafo_dir, 'TransformParameters.Similarity-3Channels.0.txt')

    interpolation = 'nearest'
    t = task(tmp_folder=tmp_folder, config_dir=conf_dir, max_jobs=1,
             input_path_file=in_file, output_path_file=out_file, transformation_file=trafo,
             interpolation=interpolation)
    ret = luigi.build([t], local_scheduler=True)
    assert ret
    expected_xml = out_path + '.xml'
    assert os.path.exists(expected_xml), expected_xml
    expected_h5 = out_path + '.h5'
    assert os.path.exists(expected_h5), expected_h5


if __name__ == '__main__':
    check_wrapper()
