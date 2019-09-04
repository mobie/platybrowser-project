from concurrent import futures
import numpy as np
import h5py
import pandas as pd
import nifty
from elf.skeleton import skeletonize
from scipy.ndimage import distance_transform_edt


def get_mapped_cell_ids(cilia_ids, manual_mapping_table_path):
    mapping_table = pd.read_csv(manual_mapping_table_path, sep='\t')

    cell_ids = np.zeros_like(cilia_ids)
    for row in mapping_table.itertuples(index=False):
        cilia_id = int(row.cilia_id)
        cell_id = int(row.cell_id)
        cell_ids[cilia_id] = cell_id

    return cell_ids


def make_indexable(path):
    return tuple(np.array([p[i] for p in path], dtype='uint64') for i in range(3))


def compute_centerline(obj, resolution):
    """ Compute the centerline path and its length
        by computing the 3d skeleton via thinning and extracting
        the longest path between terminals
    """
    # compute skeleton and graph from skeleton
    nodes, edges = skeletonize(obj)
    graph = nifty.graph.undirectedGraph(len(nodes))
    graph.insertEdges(edges)

    # compute the length of the edges
    physical_coords = nodes.astype('float32') * np.array(list(resolution))
    edge_lens = physical_coords[edges[:, 0]] - physical_coords[edges[:, 1]]
    edge_lens = np.linalg.norm(edge_lens, axis=1)
    assert edge_lens.shape == (len(edges),), str(edge_lens.shape)

    # compute the degrees and derive the terminals
    degrees = np.array([len([adj for adj in graph.nodeAdjacency(u)])
                       for u in range(graph.numberOfNodes)])
    terminals = np.where(degrees == 1)[0]
    if len(terminals) < 2:
        raise ValueError("Did not find terminals.")

    # compute length between all terminals and find the longest path
    t0, t1 = None, None
    max_plen = 0.
    sp = nifty.graph.ShortestPathDijkstra(graph)
    for ii, t in enumerate(terminals[:-1]):
        targets = terminals[ii+1:]
        paths = sp.runSingleSourceMultiTarget(edge_lens, t, targets,
                                              returnNodes=False)
        for target, p in zip(targets, paths):
            # paths can be empty, not sure why
            if not p:
                continue
            plen = edge_lens[np.array(p)].sum()
            if plen > max_plen:
                t0, t1 = t, target
                max_plen = plen

    path = sp.runSingleSourceSingleTarget(edge_lens, t0, t1,
                                          returnNodes=True)
    coordinates = make_indexable(nodes[np.array(path)])
    return coordinates, max_plen


def get_bb(base_table, cid, resolution):
    # get the row for this cilia id
    row = base_table.loc[cid]
    # compute the bounding box
    bb_min = (row.bb_min_z, row.bb_min_y, row.bb_min_x)
    bb_max = (row.bb_max_z, row.bb_max_y, row.bb_max_x)
    bb = tuple(slice(int(mi / re), int(ma / re))
               for mi, ma, re in zip(bb_min, bb_max, resolution))
    return bb


def load_seg(ds, base_table, cid, resolution):
    # load segmentation from the bounding box and get foreground
    bb = get_bb(base_table, cid, resolution)
    obj = ds[bb] == cid
    return obj


def measure_cilia_attributes(seg_path, seg_key, base_table, resolution):
    n_features = 3
    attributes = np.zeros((len(base_table), n_features), dtype='float32')
    names = ['length', 'diameter_mean', 'diameter_std']

    ids = base_table['label_id'].values.astype('uint64')

    with h5py.File(seg_path, 'r') as f:
        ds = f[seg_key]

        def compute_attributes(cid):

            # FIXME current 1 and 2 should be part of bg label
            if cid in (1, 2):
                return

            obj = load_seg(ds, base_table, cid, resolution)
            if(obj.sum() == 0):
                print("Did not find any pixels for cilia", cid)
                return

            # compute len in microns (via shortest path)
            # and diameter (via mean boundary distance transform)
            # we switch to nanometer resolution and convert back to microns later
            skel_res = [res * 1000 for res in resolution]
            try:
                path, dist = compute_centerline(obj, skel_res)
            except ValueError:
                print("Centerline computation for", cid, "failed")
                return
            dist /= 1000.

            # make path index-able
            boundary_distances = distance_transform_edt(obj, sampling=skel_res)
            diameters = boundary_distances[path]
            # we compute the radii before, so we only
            # divide by 500 (2 / 1000) to get to the diameters in micron
            diameters /= 500.
            attributes[cid, 0] = dist
            attributes[cid, 1] = np.mean(diameters)
            attributes[cid, 2] = np.std(diameters)

        n_threads = 16
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(compute_attributes, cid) for cid in ids[1:]]
            [t.result() for t in tasks]

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
