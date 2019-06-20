import os
import csv
import luigi
import z5py

from cluster_tools.node_labels import NodeLabelWorkflow


def write_csv(output_path, data, col_names):
    assert data.shape[1] == len(col_names), "%i %i" % (data.shape[1],
                                                       len(col_names))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(col_names)
        writer.writerows(data)


def node_labels(seg_path, seg_key,
                input_path, input_key, prefix,
                tmp_folder, target, max_jobs):
    task = NodeLabelWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    out_path = os.path.join(tmp_folder, 'data.n5')
    out_key = 'node_labels_%s' % prefix

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             ws_path=seg_path, ws_key=seg_key,
             input_path=input_path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix=prefix, )
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Node labels for %s" % prefix)

    with z5py.File(out_path, 'r') as f:
        data = f[out_key][:]
    return data
