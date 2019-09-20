import os
import json

import luigi
# TODO remove path hack once we merge this into master
import sys
sys.path.insert(0, '../..')
from scripts.extension.registration import ApplyRegistrationLocal


def check_wrapper_simple():
    in_path = '/g/almf/software/elastix-test/muscles.tif'

    tmp_folder = '/g/kreshuk/pape/Work/my_projects/dev-platy/test/registration/tmp_registration_simple'
    out_path = os.path.join(tmp_folder, 'out')

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

    trafo = '/g/almf/software/elastix-test/TransformParameters.RotationPreAlign.0.txt'
    t = task(tmp_folder=tmp_folder, config_dir=conf_dir, max_jobs=1,
             input_path_file=in_file, output_path_file=out_file, transformation_file=trafo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret
    expected_out_xml = out_path + '.xml'
    assert os.path.exists(expected_out_xml), expected_out_xml
    expected_out_h5 = out_path + '.h5'
    assert os.path.exists(expected_out_h5), expected_out_h5


def check_wrapper():
    in_path = os.path.join('/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/9.9.9/images/ProSPr',
                           'Stomach_forRegistration.tif')
    out_path = '/g/kreshuk/pape/Work/my_projects/dev-platy/test/registration/somach_prospr_registered'

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

    # TODO which one is the correct trafo ?
    trafo = os.path.join('/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/0.0.0/transformations/0.0.0',
                         'TransformParameters.BSpline10-3Channels.0.txt')
    t = task(tmp_folder='tmp_registration', config_dir=conf_dir, max_jobs=1,
             input_path_file=in_file, output_path_file=out_file, transformation_file=trafo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret
    expected_out = out_path + '.xml'
    assert os.path.exists(expected_out), expected_out


# check_wrapper()
check_wrapper_simple()
