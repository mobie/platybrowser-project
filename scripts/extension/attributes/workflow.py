import os
import pandas as pd
import luigi
from cluster_tools.cluster_tasks import WorkflowBase

from . import morphology as morpho_tasks


class MergeTables(luigi.Task):
    output_prefix = luigi.Parameter()
    output_path = luigi.Parameter()
    max_jobs = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        # load all job sub results
        tables = []
        for job_id in range(self.max_jobs):
            path = self.output_prefix + '_job%i.csv' % job_id
            # NOTE: not all jobs might have been scheduled, so
            # we neeed to check if the result actually exists
            if not os.path.exists(path):
                continue
            sub_table = pd.read_csv(path, sep='\t')
            tables.append(sub_table)

        table = pd.concat(tables)
        table.sort_values('label_id', inplace=True)
        table.to_csv(self.output_path, index=False, sep='\t')

    def output(self):
        return luigi.LocalTarget(self.output_path)


class MorphologyWorkflow(WorkflowBase):
    # input volumes and graph
    segmentation_path = luigi.Parameter()
    in_table_path = luigi.Parameter()
    output_path = luigi.Parameter()
    # resolution of the segmentation at full scale
    resolution = luigi.ListParameter()
    # scales of segmentation and raw data used for the computation
    seg_scale = luigi.IntParameter()
    raw_scale = luigi.IntParameter(default=3)
    # prefix
    prefix = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    # minimum and maximum sizes for objects
    min_size = luigi.IntParameter()
    max_size = luigi.IntParameter(default=None)
    # path for cell nucleus mapping, that is used for additional
    # table filtering
    mapping_path = luigi.IntParameter(default='')
    # input path for intensity calcuation
    # if '', intensities will not be calculated
    raw_path = luigi.Parameter(default='')

    def requires(self):
        out_prefix = os.path.join(self.tmp_folder, 'sub_table_%s' % self.prefix)
        morpho_task = getattr(morpho_tasks,
                              self._get_task_name('Morphology'))
        dep = morpho_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          dependency=self.dependency, max_jobs=self.max_jobs,
                          segmentation_path=self.segmentation_path,
                          in_table_path=self.in_table_path, output_prefix=out_prefix,
                          resolution=self.resolution, seg_scale=self.seg_scale, raw_scale=self.raw_scale,
                          prefix=self.prefix, number_of_labels=self.number_of_labels,
                          min_size=self.min_size, max_size=self.max_size, mapping_path=self.mapping_path,
                          raw_path=self.raw_path)
        dep = MergeTables(output_prefix=out_prefix, output_path=self.output_path,
                          max_jobs=self.max_jobs, dependency=dep)

        return dep

    @staticmethod
    def get_config():
        configs = super(MorphologyWorkflow, MorphologyWorkflow).get_config()
        configs.update({'morphology': morpho_tasks.MorphologyLocal.default_task_config()})
        return configs
