import os
import h5py
import numpy as np
import pandas as pd
import vigra

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from pybdv import make_bdv


def compute_labels(root):
    table = os.path.join(root, 'sbem-6dpf-1-whole-segmented-cells-labels', 'regions.csv')
    table = pd.read_csv(table, sep='\t')
    label_ids = table['label_id'].values
    labels = table['muscle'].values.astype('uint8')
    assert np.array_equal(np.unique(labels), [0, 1])
    return labels, label_ids


def compute_features(root):

    feature_names = []

    # we take the number of pixels and calculat the size of
    # the bounding bpx from the default table
    default = os.path.join(root, 'sbem-6dpf-1-whole-segmented-cells-labels', 'default.csv')
    default = pd.read_csv(default, sep='\t')

    n_pixels = default['n_pixels'].values
    bb_min = np.array([default['bb_min_z'].values,
                       default['bb_min_y'].values,
                       default['bb_min_x'].values]).T
    bb_max = np.array([default['bb_max_z'].values,
                       default['bb_max_y'].values,
                       default['bb_max_x'].values]).T
    bb_shape = bb_max - bb_min
    features_def = np.concatenate([n_pixels[:, None], bb_shape], axis=1)
    feature_names.extend(['n_pixels', 'bb_shape_z', 'bb_shape_y', 'bb_shape_x'])

    morpho = os.path.join(root, 'sbem-6dpf-1-whole-segmented-cells-labels', 'morphology.csv')
    morpho = pd.read_csv(morpho, sep='\t')
    label_ids_morpho = morpho['label_id'].values.astype('uint64')
    features_morpho = morpho[['shape_volume_in_microns', 'shape_extent',
                              'shape_surface_area', 'shape_sphericity']].values
    feature_names.extend(['volume', 'extent', 'surface_area', 'sphericty'])

    # add the nucleus features
    nucleus_mapping = os.path.join(root, 'sbem-6dpf-1-whole-segmented-cells-labels',
                                   'cells_to_nuclei.csv')
    nucleus_mapping = pd.read_csv(nucleus_mapping, sep='\t')
    label_ids_nuclei = nucleus_mapping['label_id'].values.astype('uint64')
    nucleus_ids = nucleus_mapping['nucleus_id'].values.astype('uint64')
    nucleus_mask = nucleus_ids > 0
    nucleus_ids = nucleus_ids[nucleus_mask]
    label_ids_nuclei = label_ids_nuclei[nucleus_mask]

    nucleus_features = os.path.join(root, 'sbem-6dpf-1-whole-segmented-nuclei-labels',
                                    'morphology.csv')
    nucleus_features = pd.read_csv(nucleus_features, sep='\t')
    nucleus_features.set_index('label_id', inplace=True)
    nucleus_features = nucleus_features.loc[nucleus_ids]
    nucleus_features = nucleus_features[['shape_volume_in_microns', 'shape_extent',
                                         'shape_surface_area', 'shape_sphericity',
                                         'intensity_mean', 'intensity_st_dev']].values
    feature_names.extend(['nucleus_volume', 'nucleus_extent', 'nucleus_surface_area',
                          'nucleus_sphericity', 'nucleus_intensity_mean', 'nucleus_intensity_std'])

    # combine the features
    label_id_mask = np.zeros(len(default), dtype='uint8')
    label_id_mask[label_ids_morpho] += 1
    label_id_mask[label_ids_nuclei] += 1
    label_id_mask = label_id_mask > 1
    valid_label_ids = np.where(label_id_mask)[0]

    label_id_mask_morpho = np.isin(label_ids_morpho, valid_label_ids)
    label_id_mask_nuclei = np.isin(label_ids_nuclei, valid_label_ids)

    features = np.concatenate([features_def[label_id_mask],
                               features_morpho[label_id_mask_morpho],
                               nucleus_features[label_id_mask_nuclei]], axis=1)
    assert len(features) == len(valid_label_ids), "%i, %i" % (len(features),
                                                              len(valid_label_ids))
    return features, valid_label_ids, feature_names


def compute_features_and_labels(root):
    features, label_ids, _ = compute_features(root)
    labels, _ = compute_labels(root)

    labels = labels[label_ids]
    assert len(labels) == len(features) == len(label_ids)
    return features, labels, label_ids


def predict_muscle_mapping(root, n_folds=5, threshold=.5):
    print("Computig labels and features ...")
    features, labels, label_ids = compute_features_and_labels(root)
    print("Found", len(features), "samples and", features.shape[1], "features per sample")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    false_pos_ids = []
    false_neg_ids = []

    print("Find false positives and negatives on", n_folds, "folds")
    for train_idx, test_idx in kf.split(features, labels):
        x, y = features[train_idx], labels[train_idx]
        rf = RandomForestClassifier(n_estimators=50, n_jobs=8)
        rf.fit(x, y)

        x, y = features[test_idx], labels[test_idx]
        # allow adapting the threshold ?
        pred = (rf.predict_proba(x)[:, 1] > threshold)

        false_pos = np.logical_and(pred == 1, y == 0)
        false_neg = np.logical_and(pred == 0, y == 1)

        fold_labels = label_ids[test_idx]
        false_pos_ids.extend(fold_labels[false_pos].tolist())
        false_neg_ids.extend(fold_labels[false_neg].tolist())

    print("Found", len(false_pos_ids), "false positves")
    print("Found", len(false_neg_ids), "false negatitves")
    return false_pos_ids, false_neg_ids


def map_ids_to_seg(seg, ids):
    bg_mask = ~np.isin(seg, ids)
    seg[bg_mask] = 0
    return seg.astype('uint32')


def ranked_false_positives(root, n_folds=5, threshold=.5):
    # compute the candidates for missing muscles
    missing_candidates, _ = predict_muscle_mapping(root, n_folds, threshold)

    # load the muscle segmentation
    path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-segmented-muscle.h5'
    muscle_scale = 2
    # muscle_scale = 3
    print("Load muscles ...")
    with h5py.File(path) as f:
        ds = f['t00000/s00/%i/cells' % muscle_scale]
        muscles = ds[:]

    # load the segmentation and map fp ids to it
    path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.3.1/segmentations',
                        'sbem-6dpf-1-whole-segmented-cells-labels.h5')
    print("Load and project seg ...")
    seg_scale = muscle_scale + 2
    with h5py.File(path) as f:
        ds = f['t00000/s00/%i/cells' % seg_scale]
        seg = ds[:]
    missing_seg = map_ids_to_seg(seg, missing_candidates)

    print("Resize muslce segmentation ...")
    muscles = vigra.sampling.resize(muscles.astype('float32'), order=0, shape=missing_seg.shape)

    print("Comute distance transform ...")
    # compute distance trafo of muscle segmentation
    distances = vigra.filters.distanceTransform((muscles > 0).astype('uint32'))

    # accumulate distances over the fp segments
    print("Accumulate distances ...")
    mean_distances = vigra.analysis.extractRegionFeatures(distances, missing_seg, features=['mean'])['mean']
    mean_distances = mean_distances[missing_candidates]
    assert len(mean_distances) == len(missing_candidates)

    return missing_candidates, mean_distances


def predict_and_save_to_h5(root, project_path, n_folds=5, threshold=.5):
    # save false pos and false negatives
    print("Serializing results to", project_path)
    false_pos_ids, false_neg_ids = predict_muscle_mapping(root, n_folds, threshold)
    with h5py.File(project_path) as f:
        f.create_dataset('false_positives/prediction', data=false_pos_ids)
        f.create_dataset('false_negatives/prediction', data=false_neg_ids)


def predict_and_save_to_csv(root, output_path, n_folds=5, threshold=.5):
    # save false pos and false negatives
    print("Serializing results to", output_path)
    false_pos_ids, false_neg_ids = predict_muscle_mapping(root, n_folds, threshold)
    labels, label_ids = compute_labels(root)

    false_pos = np.zeros_like(labels)
    false_pos[false_pos_ids] = 1

    false_neg = np.zeros_like(labels)
    false_neg[false_neg_ids] = 1

    table = np.concatenate([label_ids[:, None], labels[:, None], false_pos[:, None], false_neg[:, None]], axis=1)
    table = pd.DataFrame(data=table, columns=['label_id', 'muscle', 'potential_miss', 'potential_extra'])
    table.to_csv(output_path, sep='\t', index=False)


def get_mapped_ids(ids, out_path, scale=4):

    resolution = [0.025, 0.02, 0.02]
    resolution = [res * 2**scale for res in resolution]

    # load the segmentation and map fp ids to it
    path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.3.1/segmentations',
                        'sbem-6dpf-1-whole-segmented-cells-labels.h5')
    print("Load and project seg ...")
    with h5py.File(path, 'r') as f:
        ds = f['t00000/s00/%i/cells' % scale]
        seg = ds[:]

    seg = map_ids_to_seg(seg, ids)

    ds_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    make_bdv(seg, out_path, downscale_factors=ds_factors,
             resolution=resolution, unit='micrometer')
