import os
import csv
import luigi
import z5py
import nifty.distributed as ndist

from cluster_tools.node_labels import NodeLabelWorkflow


def write_csv(output_path, data, col_names):
    assert data.shape[1] == len(col_names), "%i %i" % (data.shape[1],
                                                       len(col_names))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(col_names)
        writer.writerows(data)


def normalize_overlap_dict(label_overlap_dict):
    sums = {label_id: sum(overlaps.values())
            for label_id, overlaps in label_overlap_dict.items()}
    label_overlap_dict = {label_id: {ovlp_id: count / sums[label_id]
                                     for ovlp_id, count in overlaps.items()}
                          for label_id, overlaps in label_overlap_dict.items()}
    return label_overlap_dict


def node_labels(seg_path, seg_key,
                input_path, input_key, prefix,
                tmp_folder, target, max_jobs,
                max_overlap=True):
    task = NodeLabelWorkflow
    config_folder = os.path.join(tmp_folder, 'configs')

    out_path = os.path.join(tmp_folder, 'data.n5')
    out_key = 'node_labels_%s' % prefix

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             ws_path=seg_path, ws_key=seg_key,
             input_path=input_path, input_key=input_key,
             output_path=out_path, output_key=out_key,
             prefix=prefix, max_overlap=max_overlap)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Node labels for %s" % prefix)

    f = z5py.File(out_path, 'r')
    ds_out = f[out_key]

    if max_overlap:
        data = ds_out[:]
    else:
        n_chunks = ds_out.number_of_chunks
        out_path = ds_out.path
        data = [ndist.deserializeOverlapChunk(out_path, (chunk_id,))[0]
                for chunk_id in range(n_chunks)]
        data = {label_id: overlaps
                for chunk_data in data
                for label_id, overlaps in chunk_data.items()}

    return data
