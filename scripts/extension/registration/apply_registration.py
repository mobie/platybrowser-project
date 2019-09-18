#! /usr/bin/python

import os
import json
import sys
from subprocess import check_output, CalledProcessError

import luigi
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class ApplyRegistrationBase(luigi.Task):
    """ ApplyRegistration base class
    """

    task_name = 'apply_registration'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path_file = luigi.Parameter()
    output_path_file = luigi.Parameter()
    transformation_file = luigi.Parameter()
    fiji_executable = luigi.Parameter(default='/g/almf/software/Fiji.app/ImageJ-linux64')
    elastix_directory = luigi.Parameter(default='/g/almf/software/elastix_v4.8')
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        with open(self.input_path_file) as f:
            inputs = json.load(f)
        with open(self.output_path_file) as f:
            outputs = json.load(f)

        assert len(inputs) == len(outputs)
        assert all(os.path.exists(inp) for inp in inputs)
        n_files = len(inputs)

        assert os.path.exists(self.transformation_file)
        assert os.path.exists(self.fiji_executable)
        assert os.path.exists(self.elastix_directory)

        # get the split of file-ids to the volume
        file_list = vu.blocks_in_volume((n_files,), (1,))

        # we don't need any additional config besides the paths
        config = {"input_path_file": self.input_path_file,
                  "output_path_file": self.output_path_file,
                  "transformation_file": self.transformation_file,
                  "fiji_executable": self.fiji_executable,
                  "elastix_directory": self.elastix_directory,
                  "tmp_folder": self.tmp_folder}

        # prime and run the jobs
        n_jobs = min(self.max_jobs, n_files)
        self.prepare_jobs(n_jobs, file_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class ApplyRegistrationLocal(ApplyRegistrationBase, LocalTask):
    """
    ApplyRegistration on local machine
    """
    pass


class ApplyRegistrationSlurm(ApplyRegistrationBase, SlurmTask):
    """
    ApplyRegistration on slurm cluster
    """
    pass


class ApplyRegistrationLSF(ApplyRegistrationBase, LSFTask):
    """
    ApplyRegistration on lsf cluster
    """
    pass


#
# Implementation
#

def apply_for_file(input_path, output_path,
                   transformation_file, fiji_executable,
                   elastix_directory, tmp_folder, n_threads):

    # grrr design anti-pattern
    # the argument to transformaix needs to be one large comma separated string
    transformix_argument = ["elastix_directory=\'%s\'" % elastix_directory,
                            "workingDirectory=\'%s\'" % tmp_folder,
                            "inputImageFile=\'%s\'" % input_path,
                            "transformationFile=\'%s\'" % transformation_file,
                            "outputFile=\'%s\'" % output_path,
                            "outputModality=\'Save as BigDataViewer .xml/.h5\'",
                            "numThreads=\'1\'"]  # TODO why do we use numThreads=1 and not the same as in -c?
    transformix_argument = ",".join(transformix_argument)
    transformix_argument = "\"%s\"" % transformix_argument

    # command based on https://github.com/embl-cba/fiji-plugin-elastixWrapper/issues/2:
    # srun --mem 16000 -n 1 -N 1 -c 8 -t 30:00 -o $OUT -e $ERR
    # /g/almf/software/Fiji.app/ImageJ-linux64  --ij2 --headless --run "Transformix"
    # "elastixDirectory='/g/almf/software/elastix_v4.8', workingDirectory='$TMPDIR',
    # inputImageFile='$INPUT_IMAGE',transformationFile='/g/cba/exchange/platy-trafos/linear/TransformParameters.BSpline10-3Channels.0.txt
    # outputFile='$OUTPUT_IMAGE',outputModality='Save as BigDataViewer .xml/.h5',numThreads='1'"
    # NOTE: I ommit --run here, because fiji throws a warning that it does not recognise the argument
    cmd = [fiji_executable, "--ij2", "--headless", "\"Transformix\"", transformix_argument]

    cmd_str = " ".join(cmd)
    fu.log("Calling the following command:")
    fu.log(cmd_str)

    try:
        check_output(cmd)
    except CalledProcessError as e:
        raise RuntimeError(e.output)


def apply_registration(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)

    # get list of the input and output paths
    input_file = config['input_path_file']
    with open(input_file) as f:
        inputs = json.load(f)
    output_file = config['output_path_file']
    with open(output_file) as f:
        outputs = json.load(f)

    transformation_file = config['transformation_file']
    fiji_executable = config['fiji_executable']
    elastix_directory = config['elastix_directory']
    tmp_folder = config['tmp_folder']
    working_dir = os.path.join(tmp_folder, 'work_dir%i' % job_id)
    os.makedirs(working_dir, exist_ok=True)

    file_list = config['block_list']
    n_threads = config.get('threads_per_job', 1)

    fu.log("Applying registration with:")
    fu.log("transformation_file: %s" % transformation_file)
    fu.log("fiji_executable: %s" % fiji_executable)
    fu.log("elastix_directory: %s" % elastix_directory)

    for file_id in file_list:
        fu.log("start processing block %i" % file_id)

        infile = inputs[file_id]
        outfile = outputs[file_id]
        fu.log("Input: %s" % infile)
        fu.log("Output: %s" % outfile)
        apply_for_file(infile, outfile,
                       transformation_file, fiji_executable,
                       elastix_directory, working_dir, n_threads)
        fu.log_block_success(file_id)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    apply_registration(job_id, path)
