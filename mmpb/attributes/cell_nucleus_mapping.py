import numpy as np
import h5py
from .util import write_csv, node_labels


def overlaps_to_ids(overlaps, overlap_threshold):
    # normalize the overlaps
    overlap_counts = {label_id: float(sum(ovlps.values())) for label_id, ovlps in overlaps.items()}
    # return all overlaps that are consistent with the overlap threshold
    overlaps = {label_id: [ovlp_id for ovlp_id, ovlp in ovlps.items()
                           if ((ovlp / overlap_counts[label_id]) > overlap_threshold) and (ovlp_id != 0)]
                for label_id, ovlps in overlaps.items()}
    return overlaps


# TODO enable serializing the whole thing, because we might want to use this
# for guided proof-reading

# TODO do we need size filtering?
# (this would imply doing it after we have nucleus tables as well!)
def map_cells_to_nuclei(label_ids, seg_path, nuc_path, out_path,
                        tmp_folder, target, max_jobs,
                        overlap_threshold=.25):

    # choose the keys of the same size
    seg_key = 't00000/s00/2/cells'
    nuc_key = 't00000/s00/0/cells'
    with h5py.File(seg_path, 'r') as f:
        shape1 = f[seg_key].shape
    with h5py.File(nuc_path, 'r') as f:
        shape2 = f[nuc_key].shape
    assert shape1 == shape2

    # TODO try  setting 0 to ignore label (with higher overlap threshold?)
    # compute the pixel-wise overlap of cells with nuclei
    cids_to_nids = node_labels(seg_path, seg_key,
                               nuc_path, nuc_key, prefix='nuc_to_cells',
                               tmp_folder=tmp_folder, target=target, max_jobs=max_jobs,
                               max_overlap=False, ignore_label=0)
    cids_to_nids = overlaps_to_ids(cids_to_nids, overlap_threshold)

    # compute the pixel-wise overlap of nuclei with cells
    nids_to_cids = node_labels(nuc_path, nuc_key,
                               seg_path, seg_key, prefix='cells_to_nuc',
                               tmp_folder=tmp_folder, target=target, max_jobs=max_jobs,
                               max_overlap=False, ignore_label=0)
    nids_to_cids = overlaps_to_ids(nids_to_cids, overlap_threshold)

    # only keep cell ids that have overlap with a single nucleus
    cids_to_nids = {label_id: ovlp_ids[0] for label_id, ovlp_ids in cids_to_nids.items()
                    if len(ovlp_ids) == 1}

    # only keep nucleus ids that have overlap with a single cell
    nids_to_cids = {label_id: ovlp_ids[0] for label_id, ovlp_ids in nids_to_cids.items()
                    if len(ovlp_ids) == 1}

    # only keep cell ids for which overlap-ids agree
    cids_to_nids = {label_id: ovlp_id for label_id, ovlp_id in cids_to_nids.items()
                    if nids_to_cids.get(ovlp_id, 0) == label_id}

    data = np.array([cids_to_nids.get(label_id, 0) for label_id in label_ids])

    col_names = ['label_id', 'nucleus_id']
    data = np.concatenate([label_ids[:, None], data[:, None]], axis=1)
    write_csv(out_path, data, col_names)
