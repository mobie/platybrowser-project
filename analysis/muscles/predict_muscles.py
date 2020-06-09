import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def get_features_and_labels(version):
    feature_path = f'../../data/{version}/tables/sbem-6dpf-1-whole-segmented-cells/morphology.csv'
    labels_path = f'../../data/{version}/tables/sbem-6dpf-1-whole-segmented-cells/regions.csv'

    features = pd.read_csv(feature_path, sep='\t').values
    cell_ids, features = features[:, 0].astype('uint32'), features[:, 1:]

    labels = pd.read_csv(labels_path, sep='\t')
    label_cell_ids = labels['label_id'].values.astype('uint32')
    # make sure we have labels for all the cell ids
    assert np.setdiff1d(cell_ids, label_cell_ids).size == 0

    labels = labels['muscle'].values.astype('uint8')
    label_mask = np.isin(label_cell_ids, cell_ids)
    labels = labels[label_mask]

    assert len(features) == len(labels) == len(cell_ids)
    return features, labels, cell_ids


def predict_muscles(version, save_path, n_folds=5, preserve_pos_labels=True):
    features, labels, cell_ids = get_features_and_labels(version)

    skf = StratifiedKFold(n_splits=n_folds)
    muscle_predictions = np.zeros_like(labels)

    n_jobs = 32
    n_trees = 100

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print("Processing fold", fold_id, "/", n_folds)

        x_train, y_train = features[train_idx], labels[train_idx]
        rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs)
        rf.fit(x_train, y_train)

        x_test, y_test = features[test_idx], labels[test_idx]
        pred = rf.predict(x_test)
        assert pred.shape == y_test.shape

        print("predicted", pred.sum(), "/", pred.size, "cells as muscle")
        if preserve_pos_labels:
            pred[y_test == 1] = 1
            print("after preserving positive labels:", pred.sum(), "/", pred.size)

        muscle_predictions[test_idx] = pred

    assert len(cell_ids) == len(muscle_predictions)
    tab = np.concatenate([cell_ids[:, None], muscle_predictions[:, None]], axis=1)
    tab = pd.DataFrame(tab, columns=['label_id', 'muscle'])
    tab.to_csv(save_path, index=False, sep='\t')


if __name__ == '__main__':
    version = '1.0.1'
    save_path = './muscle_predictions.csv'
    predict_muscles(version, save_path)
