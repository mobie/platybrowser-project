import os
import luigi
import z5py
import nifty.tools as nt

from cluster_tools.cluster_tasks import WorkflowBase
from cluster_tools.node_labels import NodeLabelWorkflow
from cluster_tools.relabel import RelabelWorkflow
from cluster_tools.postprocess import filter_blocks as filter_tasks

from . import map_nuclei as map_tasks


class RelabelAssignments(luigi.Task):
    tmp_folder = luigi.Parameter()
    path = luigi.Parameter()
    key = luigi.Parameter()
    relabeling = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        f = z5py.File(self.path)
        ds = f[self.key]
        assignments = ds[:]
        relabeling = f[self.relabeling][:]
        relabeling = dict(zip(relabeling[:, 0], relabeling[:, 1]))

        to_relabel = assignments[:, 0]
        relabeled = nt.takeDict(relabeling, to_relabel)
        assignments[:, 0] = relabeled
        ds[:] = assignments

        with open(self.output().path, 'w') as log:
            log.write("Success")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'relabel_assignments.log'))


class NucleusAssignmentWorkflow(WorkflowBase):
    path = luigi.Parameter()
    seg_key = luigi.Parameter()
    nucleus_seg_key = luigi.Parameter()
    prefix = luigi.Parameter()
    output_key = luigi.Parameter(default=None)
    relabel = luigi.BoolParameter(default=True)
    max_overlap = luigi.BoolParameter(default=False)

    def requires(self):
        # find overlaps with nuclei
        tmp_key = 'nuclei_overlaps/%s' % self.prefix
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                target=self.target, config_dir=self.config_dir,
                                input_path=self.path, input_key=self.nucleus_seg_key,
                                ws_path=self.path, ws_key=self.seg_key,
                                output_path=self.path, output_key=tmp_key,
                                prefix=self.prefix, max_overlap=self.max_overlap,
                                ignore_label=0, serialize_counts=True)

        if self.output_key is not None:
            assert False, "Needs update"
            with z5py.File(self.path) as f:
                n_labels = int(f[self.seg_key].attrs['maxId']) + 1

            assignment_key = 'nuclei_overlaps/assignments_%s' % self.prefix
            filter_path = os.path.join(self.tmp_folder, 'filtered_ids.json')
            map_task = getattr(map_tasks, self._get_task_name('MapNuclei'))
            dep = map_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                           config_dir=self.config_dir, dependency=dep,
                           path=self.path, key=tmp_key,
                           res_path=filter_path, output_key=assignment_key,
                           n_labels=n_labels)

            filter_task = getattr(filter_tasks, self._get_task_name('FilterBlocks'))
            dep = filter_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                              dependency=dep, max_jobs=self.max_jobs,
                              input_path=self.path, input_key=self.seg_key,
                              filter_path=filter_path,
                              output_path=self.path, output_key=self.output_key)

            if self.relabel:
                relabel_key = 'relabel_nucleus_filter_%s' % self.prefix
                dep = RelabelWorkflow(tmp_folder=self.tmp_folder,
                                      max_jobs=self.max_jobs,
                                      config_dir=self.config_dir,
                                      target=self.target,
                                      input_path=self.path,
                                      input_key=self.output_key,
                                      assignment_path=self.path,
                                      assignment_key=relabel_key,
                                      dependency=dep)
                # relabel first column of the assignments
                dep = RelabelAssignments(tmp_folder=self.tmp_folder, path=self.path, key=assignment_key,
                                         relabeling=relabel_key, dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(NucleusAssignmentWorkflow, NucleusAssignmentWorkflow).get_config()
        configs.update({'map_nuclei': map_tasks.MapNucleiLocal.default_task_config(),
                        'filter_blocks': filter_tasks.FilterBlocksLocal.default_task_config(),
                        **RelabelWorkflow.get_config()})
        return configs
