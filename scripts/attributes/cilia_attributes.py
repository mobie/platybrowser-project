from concurrent import futures
import numpy as np
import h5py
import pandas as pd
from elf.skeleton.teasar import Teasar


def get_mapped_cell_ids(cilia_ids, manual_mapping_table_path):
    mapping_table = pd.read_csv(manual_mapping_table_path, sep='\t')

    cell_ids = np.zeros_like(cilia_ids)
    for row in mapping_table.itertuples(index=False):
        cilia_id = int(row.cilia_id)
        cell_id = int(row.cell_id)
        cell_ids[cilia_id] = cell_id

    return cell_ids


def measure_cilia_attributes(seg_path, seg_key, base_table, resolution):
    n_features = 2
    attributes = np.zeros((len(base_table), n_features), dtype='float32')
    names = ['length', 'diameter']

    ids = base_table['label_id'].values.astype('uint64')

    with h5py.File(seg_path, 'r') as f:
        ds = f[seg_key]

        def compute_attributes(cid):

            # FIXME 1 and 2 should be part of bg label
            if cid in (1, 2):
                return

            # get the row for this cilia id
            row = base_table.loc[cid]

            # compute the bounding box
            bb_min = (row.bb_min_z, row.bb_min_y, row.bb_min_x)
            bb_max = (row.bb_max_z, row.bb_max_y, row.bb_max_x)
            bb = tuple(slice(int(mi / re), int(ma / re))
                       for mi, ma, re in zip(bb_min, bb_max, resolution))

            # load segmentation from the bounding box and get foreground
            obj = ds[bb] == cid

            # compute len in microns (via shortest path) and diameter (via mean boundary distance transform)
            teasar = Teasar(obj, resolution)
            src = teasar.root_node
            target = np.argmax(teasar.penalized_distances)
            path = teasar.get_path(src, target)
            dist = teasar.get_pathlength(path)
            print(path[:5])
            print(dist)
            # diameters = teasar.boundary_distances[path]
            # diameter = np.mean(diameters)
            attributes[cid, 0] = dist
            # attributes[cid, 1] = diameter

        [compute_attributes(cid) for cid in ids[1:]]
        # n_threads = 8
        # with futures.ThreadPoolExecutor(n_threads) as tp:
        #     tasks = [tp.submit(compute_attributes(cid)) for cid in ids[1:]]
        #     [t.result() for t in tasks]

    return attributes, names


# TODO the cell id mapping table should be separate
# TODO wrap this into a luigi task so we don't recompute it every time
def cilia_attributes(seg_path, seg_key,
                     base_table_path, manual_mapping_table_path, table_out_path,
                     resolution, tmp_folder, target, max_jobs):

    # read the base table
    base_table = pd.read_csv(base_table_path, sep='\t')
    cilia_ids = base_table['label_id'].values.astype('uint64')

    # add the manually mapped cell ids
    cell_ids = get_mapped_cell_ids(cilia_ids, manual_mapping_table_path)
    assert len(cell_ids) == len(cilia_ids)

    # measure cilia specific attributes: length, diameter, ? (could try curvature)
    attributes, names = measure_cilia_attributes(seg_path, seg_key, base_table, resolution)
    assert len(attributes) == len(cilia_ids)
    assert attributes.shape[1] == len(names)

    table = np.concatenate([cilia_ids[:, None], cell_ids[:, None], attributes], axis=1)
    col_names = ['label_id', 'cell_id'] + names
    table = pd.DataFrame(table, columns=col_names)
    table.to_csv(table_out_path, index=False, sep='\t')
