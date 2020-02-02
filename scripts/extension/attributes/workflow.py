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
    # compute cell features or nucleus features?
    compute_cell_features = luigi.BoolParameter()

    # paths to raw data and segmentations
    # if the raw path is None, we don't compute intensity features
    raw_path = luigi.Parameter(default=None)
    # we always need the nucleus segmentation
    nucleus_segmentation_path = luigi.Paramter()
    # we only need the cell segmentation if we compute cell morphology features
    cell_segmentation_path = luigi.Parameter(default=None)
    # we only need the chromatin segmentation if we compute nucleus features
    chromatin_segmentation_path = luigi.Paramter(default=None)

    # the scale used for computation, relative to the raw scale
    scale = luigi.IntParameter(default=3)

    # the input tables paths for the default table, the
    # nucleus mapping table and the region mapping table
    in_table_path = luigi.Parameter()
    # only need the mapping paths for the nucleus features
    nucleus_mapping_path = luigi.Paramter(default=None)
    region_mapping_path = luigi.Paramter(default=None)

    # minimum and maximum sizes for objects / bounding box
    min_size = luigi.IntParameter()
    max_size = luigi.IntParameter(default=None)
    max_bb = luigi.IntParameter()

    output_path = luigi.Paramter()

    def requires(self):
        out_prefix = os.path.join(self.tmp_folder, 'sub_table_%s' % self.prefix)
        morpho_task = getattr(morpho_tasks,
                              self._get_task_name('Morphology'))
        dep = morpho_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          dependency=self.dependency, max_jobs=self.max_jobs,
                          compute_cell_features=self.compute_cell_features,
                          raw_path=self.raw_path,
                          nucleus_segmentation_path=self.nucleus_segmentation_path,
                          cell_segmentation_path=self.cell_segmentation_path,
                          chromatin_segmentation_path=self.chromatin_segmentation_path,
                          in_table_path=self.in_table_path, output_prefix=out_prefix,
                          nucleus_mapping_path=self.nucleus_mapping_path,
                          region_mapping_path=self.region_mapping_path,
                          min_size=self.min_size, max_size=self.max_size, max_bb=self.max_bb)
        dep = MergeTables(output_prefix=out_prefix, output_path=self.output_path,
                          max_jobs=self.max_jobs, dependency=dep)

        return dep

    @staticmethod
    def get_config():
        configs = super(MorphologyWorkflow, MorphologyWorkflow).get_config()
        configs.update({'morphology': morpho_tasks.MorphologyLocal.default_task_config()})
        return configs
