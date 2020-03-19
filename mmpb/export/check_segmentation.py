import os
import json

import numpy as np
import luigi
import z5py

import nifty
from cluster_tools.graph import GraphWorkflow
from cluster_tools.node_labels import NodeLabelWorkflow
from mmpb.default_config import write_default_global_config


def _cc_nifty(out_path, graph_key, node_labels, ignore_label):
    with z5py.File(out_path, 'r') as f:
        group = f[graph_key]
        ds_edges = group['edges']
        edges = ds_edges[:]

    n_nodes = int(edges.max()) + 1
    g = nifty.graph.undirectedGraph(n_nodes)
    g.insertEdges(edges)
    print("Number of nodes", g.numberOfNodes, "number of edges", g.numberOfEdges)

    ccs = nifty.graph.connectedComponentsFromNodeLabels(g, node_labels, ignoreBackground=ignore_label)
    return ccs


def compute_connected_components(ws_path, ws_key,
                                 seg_path, seg_key,
                                 out_path, node_label_key, cc_key,
                                 tmp_folder, target, max_jobs,
                                 graph_key='graph', ignore_label=True):

    config_folder = os.path.join(tmp_folder, 'configs')
    write_default_global_config(config_folder)

    #
    # compute the graph
    #
    task = GraphWorkflow
    configs = task.get_config()
    conf = configs['initial_sub_graphs']
    conf.update({'ignore_label': ignore_label})

    with open(os.path.join(config_folder, 'inital_sub_graphs.config'), 'w') as f:
        json.dump(conf, f)

    n_threads = 8
    task_names = ['merge_sub_graphs', 'map_edge_ids']
    for tt in task_names:
        conf = configs['map_edge_ids']
        conf.update({'threads_per_job': n_threads, 'mem_limit': 128})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder, target=target,
             input_path=ws_path, input_key=ws_key,
             graph_path=out_path, output_key=graph_key)

    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Graph computation failed"

    #
    # compute the node labels
    #
    task = NodeLabelWorkflow

    # configs = task.get_config()

    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             target=target, config_dir=config_folder,
             ws_path=ws_path, ws_key=ws_key,
             input_path=seg_path, input_key=seg_key,
             output_path=out_path, output_key=node_label_key,
             ignore_label=0 if ignore_label else None)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Node label computation failed"

    with z5py.File(out_path, 'r') as f:
        node_labels = f[node_label_key][:]

    #
    # load the graph and check for connected components
    #
    ccs = _cc_nifty(out_path, graph_key, node_labels, ignore_label)
    return node_labels, ccs


def check_connected_components(ws_path, ws_key,
                               seg_path, seg_key,
                               tmp_folder, target,
                               max_jobs, ignore_label=True,
                               margin=0):
    out_path = os.path.join(tmp_folder, 'data.n5')
    node_label_key = 'node_labels/node_labels'
    cc_key = 'node_labels/connected_components'

    if os.path.exists(out_path):
        f = z5py.File(out_path, 'r')
        if cc_key in f:
            print("Have node labels already")
            ds = f[node_label_key]
            node_labels = ds[:]
            ds = f[cc_key]
            ccs = ds[:]
        else:
            print("Computing node labels")
            node_labels, ccs = compute_connected_components(ws_path, ws_key,
                                                            seg_path, seg_key,
                                                            out_path, node_label_key, cc_key,
                                                            tmp_folder, target, max_jobs,
                                                            ignore_label=ignore_label)
    else:
        print("Computing node labels")
        node_labels, ccs = compute_connected_components(ws_path, ws_key,
                                                        seg_path, seg_key,
                                                        out_path, node_label_key, cc_key,
                                                        tmp_folder, target, max_jobs,
                                                        ignore_label=ignore_label)

    f = z5py.File(out_path)
    if cc_key not in f:
        f.create_dataset(cc_key, data=ccs, compression='gzip', chunks=(len(ccs),))

    node_labels = np.unique(node_labels)
    n_nodes = len(node_labels)
    components = np.unique(ccs)
    n_components = len(components)

    if abs(n_nodes - n_components) <= margin:
        return True
    else:
        print("Number of components before labeling:", n_nodes)
        print("Number of components after  labeling:", n_components)
        return False
