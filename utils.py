import os

ROOT_DIR = os.path.join(os.getcwd(), os.path.pardir)

import numpy as np
import pandas as pd

from typing import List

from sklearn.utils.class_weight import compute_class_weight


def setup_train_test_idx(X, last_train_time_step, last_time_step, aggregated_timestamp_column='time_step'):
    """ The aggregated_time_step_column needs to be a column with integer values, such as year, month or day """

    split_timesteps = {}

    split_timesteps['train'] = list(range(last_train_time_step + 1))
    split_timesteps['test'] = list(range(last_train_time_step + 1, last_time_step + 1))

    train_test_idx = {}
    train_test_idx['train'] = X[X[aggregated_timestamp_column].isin(split_timesteps['train'])].index
    train_test_idx['test'] = X[X[aggregated_timestamp_column].isin(split_timesteps['test'])].index

    return train_test_idx


def train_test_split(X, y, train_test_idx):
    X_train_df = X.loc[train_test_idx['train']]
    X_test_df = X.loc[train_test_idx['test']]

    y_train = y.loc[train_test_idx['train']]
    y_test = y.loc[train_test_idx['test']]

    return X_train_df, X_test_df, y_train, y_test


def import_elliptic_data_from_csvs(root_dataset_path: str) -> List[pd.DataFrame]:
    df_classes = pd.read_csv(os.path.join(ROOT_DIR, f'{root_dataset_path}/elliptic_txs_classes.csv'))
    df_edges = pd.read_csv(os.path.join(ROOT_DIR, f'{root_dataset_path}/elliptic_txs_edgelist.csv'))
    df_features = pd.read_csv(os.path.join(ROOT_DIR, f'{root_dataset_path}/elliptic_txs_features.csv'), header=None)
    return df_classes, df_edges, df_features


def calc_occurences_per_timestep():
    X, y = load_elliptic_data()
    X['class'] = y
    occ = X.groupby(['time_step', 'class']).size().to_frame(name='occurences').reset_index()
    return occ


def rename_classes(df_classes):
    df_classes.replace({'class': {'1': 1, '2': 0, 'unknown': 2}}, inplace=True)
    return df_classes


def rename_features(df_features):
    df_features.columns = ['id', 'time_step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(72)]
    return df_features


def import_and_clean_elliptic_data():
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_classes = rename_classes(df_classes)
    df_features = rename_features(df_features)
    return df_classes, df_edges, df_features


def combine_dataframes(df_classes, df_features, only_labeled=True):
    df_combined = pd.merge(df_features, df_classes, left_on='id', right_on='txId', how='left')
    if only_labeled:
        df_combined = df_combined[df_combined['class'] != 2].reset_index(drop=True)
    df_combined.drop(columns=['txId'], inplace=True)
    return df_combined


def import_elliptic_edgelist():
    _, df_edges, df_features = import_and_clean_elliptic_data()
    df_edgelist = df_edges.merge(df_features[['id', 'time_step']], left_on='txId1', right_on='id')
    return df_edgelist


def load_elliptic_data(only_labeled=True, drop_node_id=True):
    df_classes, _, df_features = import_elliptic_data_from_csvs()
    df_features = rename_features(df_features)
    df_classes = rename_classes(df_classes)
    df_combined = combine_dataframes(df_classes, df_features, only_labeled)

    if drop_node_id:
        X = df_combined.drop(columns=['id', 'class'])
    else:
        X = df_combined.drop(columns='class')

    y = df_combined['class']

    return X, y

def get_unbalanced_weights(labels, weight1, weight2):
    class_weight = compute_class_weight({0: weight1, 1: weight2}, classes=np.unique(labels), y=labels)
    weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
    return weights

def run_elliptic_preprocessing_pipeline(last_train_time_step, last_time_step, only_labeled=True,
                                        drop_node_id=True):
    X, y = load_elliptic_data(only_labeled, drop_node_id)
    train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

    return X_train_df, X_test_df, y_train, y_test


def split_train_val_eval(last_train_time_step, last_time_step, only_labeled=True, drop_node_id=True):
    X_train_df, X_test_df, y_train_s, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step,
                                                                                   last_time_step,
                                                                                   only_labeled,
                                                                                   drop_node_id)

    nrows, ndim = X_train_df.shape
    step = int(nrows) * 0.8
    X_train = X_train_df.loc[:step]
    y_train = y_train_s.loc[:step]

    X_val = X_train_df.loc[step + 1:]
    y_val = y_train_s.loc[step + 1:]

    return X_train, X_val, X_test_df, y_train, y_val, y_test
