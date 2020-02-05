import os
import luigi

from cluster_tools.cluster_tasks import WorkflowBase
from cluster_tools import write as write_tasks
from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow

from . import fix_merges as fix_tasks
from . import find_merges as find_tasks


class UnmergeWorkflow(WorkflowBase):
    path = luigi.Parameter()
    problem_path = luigi.Parameter()

    ws_key = luigi.Parameter()
    assignment_key = luigi.Parameter()
    nucleus_mapping_key = luigi.Parameter()

    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()
    node_label_key = luigi.Parameter()

    ass_out_key = luigi.Parameter()
    out_key = luigi.Parameter(default=None)

    # minimal overlap to count nucleus as assigned
    min_overlap = luigi.IntParameter()
    from_costs = luigi.BoolParameter(default=True)
    relabel = luigi.BoolParameter(default=True)
    # clear ids that will not be fixed (for cuticle and neuropil objs)
    clear_ids = luigi.ListParameter(default=[])
    min_size = luigi.IntParameter(default=None)

    def filter_sizes(self, dep, ):
        write_task = getattr(write_tasks, self._get_task_name('Write'))
        tmp_out_key = self.out_key + '_tmp'
        tmp_ass_key = self.ass_out_key + '_tmp'
        dep = write_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         input_path=self.path,
                         input_key=self.ws_key,
                         output_path=self.path,
                         output_key=tmp_out_key,
                         assignment_path=self.path,
                         assignment_key=tmp_ass_key,
                         identifier='unmerge-temp',
                         dependency=dep)
        dep = SizeFilterAndGraphWatershedWorkflow(tmp_folder=self.tmp_folder,
                                                  max_jobs=self.max_jobs,
                                                  config_dir=self.config_dir,
                                                  target=self.target,
                                                  dependency=dep,
                                                  problem_path=self.problem_path,
                                                  graph_key=self.graph_key,
                                                  features_key=self.features_key,
                                                  path=self.path,
                                                  segmentation_key=tmp_out_key,
                                                  fragments_key=self.ws_key,
                                                  assignment_key=tmp_ass_key,
                                                  output_path=self.path,
                                                  output_key=self.out_key,
                                                  assignment_out_key=self.ass_out_key,
                                                  size_threshold=self.min_size,
                                                  relabel=self.relabel,
                                                  from_costs=self.from_costs)
        return dep

    def requires(self):
        dep = self.dependency
        # 1.) find the objects that have more than one nucleus assigned (according to the overlap threshold)
        merge_object_path = os.path.join(self.tmp_folder, 'merge_objects.json')
        find_task = getattr(find_tasks, self._get_task_name('FindMerges'))
        dep = find_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                        dependency=dep, max_jobs=1,
                        path=self.path, key=self.nucleus_mapping_key,
                        min_overlap=self.min_overlap,
                        out_path=merge_object_path, clear_ids=self.clear_ids)

        # 2.) resolve the objects with merges using graph watershed
        fix_task = getattr(fix_tasks, self._get_task_name('FixMerges'))

        # check if we need to write to tempory (if we filter sizes later)
        if self.min_size is None:
            ass_out_key = self.ass_out_key
        else:
            ass_out_key = self.ass_out_key + '_tmp'

        dep = fix_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                       dependency=dep, max_jobs=1,
                       path=self.path, problem_path=self.problem_path, assignment_key=self.assignment_key,
                       graph_key=self.graph_key, features_key=self.features_key, node_label_key=self.node_label_key,
                       out_key=ass_out_key, from_costs=self.from_costs, relabel=self.relabel,
                       merge_object_path=merge_object_path)

        # 3.) filter sizes if specified
        if self.min_size is not None:
            dep = self.filter_sizes(dep)
        elif self.out_key is not None:
            write_task = getattr(write_tasks, self._get_task_name('Write'))
            dep = write_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.path,
                             input_key=self.ws_key,
                             output_path=self.path,
                             output_key=self.out_key,
                             assignment_path=self.path,
                             assignment_key=self.ass_out_key,
                             identifier='unmerge',
                             dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(UnmergeWorkflow, UnmergeWorkflow).get_config()
        configs.update({'fix_merges':
                        fix_tasks.FixMergesLocal.default_task_config(),
                        'find_merges':
                        find_tasks.FindMergesLocal.default_task_config(),
                        'write':
                        write_tasks.WriteLocal.default_task_config()})
        return configs
