import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from model_performance import *

ROOT_DIR = os.getcwd()

import pandas as pd



def plot_performance_per_timestep(model_metric_dict, last_train_time_step=34,last_time_step=49, model_std_dict=None, fontsize=23, labelsize=18, figsize=(20, 10),
                                  linestyle=['solid', "dotted", 'dashed'], linecolor=["green", "orange", "red"],
                                  barcolor='lightgrey', baralpha=0.3, linewidth=1.5, savefig_path=None):
    occ = calc_occurences_per_timestep()
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]

    timesteps = illicit_per_timestep['time_step'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    i = 0
    for key, values in model_metric_dict.items():
        if key != "XGBoost":
            key = key.lower()
        ax1.plot(timesteps, values, label=key, linestyle=linestyle[i], color=linecolor[i], linewidth=linewidth)
        if model_std_dict != None:
            ax1.fill_between(timesteps, values + model_std_dict[key], values - model_std_dict[key],
                             facecolor='lightgrey', alpha=0.5)
        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha, label='\# illicit')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Time step', fontsize=fontsize)
    ax1.set_ylabel('Illicit F1', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step+1,last_time_step+1))
    ax1.set_yticks([0,0.25,0.5,0.75,1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Num. samples', fontsize=fontsize)

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(os.path.join(ROOT_DIR, savefig_path), bbox_inches='tight', pad_inches=0)

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

def import_elliptic_data_from_csvs():
    df_classes = pd.read_csv(os.path.join(ROOT_DIR, 'drive/MyDrive/Research/Dataset/elliptic/elliptic_txs_classes.csv'))
    df_edges = pd.read_csv(os.path.join(ROOT_DIR, 'drive/MyDrive/Research/Dataset/elliptic/elliptic_txs_edgelist.csv'))
    df_features = pd.read_csv(os.path.join(ROOT_DIR, 'drive/MyDrive/Research/Dataset/elliptic/elliptic_txs_features.csv'), header=None)
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
    if only_labeled == True:
        df_combined = df_combined[df_combined['class'] != 2].reset_index(drop=True)
    df_combined.drop(columns=['txId'], inplace=True)
    return df_combined


def import_elliptic_edgelist():
    df_classes, df_edges, df_features = import_and_clean_elliptic_data()
    df_edgelist = df_edges.merge(df_features[['id', 'time_step']], left_on='txId1', right_on='id')
    return df_edgelist


def load_elliptic_data(only_labeled=True, drop_node_id=True):
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_features = rename_features(df_features)
    df_classes = rename_classes(df_classes)
    df_combined = combine_dataframes(df_classes, df_features, only_labeled)

    if drop_node_id == True:
        X = df_combined.drop(columns=['id', 'class'])
    else:
        X = df_combined.drop(columns='class')

    y = df_combined['class']

    return X, y


def run_elliptic_preprocessing_pipeline(last_train_time_step, last_time_step, only_labeled=True,
                                        drop_node_id=True):
    X, y = load_elliptic_data(only_labeled, drop_node_id)
    train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

    return X_train_df, X_test_df, y_train, y_test