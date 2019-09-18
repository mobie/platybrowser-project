import os
import json

import luigi
# TODO remove path hack once we merge this into master
import sys
sys.path.insert(0, '../..')
from scripts.extension.registration import ApplyRegistrationLocal


def check_wrapper():
    in_path = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/stomach_prospr_target.xml'
    out_path = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/stomach_prospr_target_registered'

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
    trafo = '/g/kreshuk/pape/Work/my_projects/platy-browser-data/registration/0.0.0/transformations/0.0.0/TransformParameters.Affine-3Channels.0.txt'
    t = task(tmp_folder='tmp_registration', config_dir=conf_dir, max_jobs=1,
             input_path_file=in_file, output_path_file=out_file, transformation_file=trafo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret
    expected_out = out_path + '.xml'
    assert os.path.exists(expected_out), expected_out


check_wrapper()
