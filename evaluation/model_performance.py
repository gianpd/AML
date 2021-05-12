import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def calculate_model_score(y_true, y_pred, metric=None):
    metric_dict = {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, pos_label=1),
                   'f1': f1_score(y_true, y_pred),
                   'f1_micro': f1_score(y_true, y_pred, average='micro'),
                   'f1_macro': f1_score(y_true, y_pred, average='macro'),
                   'precision': precision_score(y_true, y_pred),
                   'recall': recall_score(y_true, y_pred),
                   'roc_auc': roc_auc_score(y_true, y_pred)}
    return metric_dict[metric] if metric else metric_dict



def calc_model_performance_over_time(X_test_df, y_test,
                                     contamination_levels_subset, scoring='f1', aggregated_timestamp_column='time_step',
                                     **model_predictions):
    first_test_time_step = np.sort(X_test_df[aggregated_timestamp_column].unique())[0]
    last_time_step = np.sort(X_test_df[aggregated_timestamp_column].unique())[-1]

    model_scores_dict = {scoring: {key: {} for key in contamination_levels_subset}}
    for contamination_level in contamination_levels_subset:
        for model_name, predictions in model_predictions.items():
            model_scores = []
            y_pred_ = predictions[contamination_level]
            for time_step in range(first_test_time_step, last_time_step + 1):
                time_step_idx = np.flatnonzero(X_test_df[aggregated_timestamp_column] == time_step)
                y_true = y_test[X_test_df[aggregated_timestamp_column] == time_step]
                y_pred = [y_pred_[i] for i in time_step_idx]

                model_scores.append(calculate_model_score(y_true.astype('int'), y_pred, scoring))
            model_scores_dict[scoring][contamination_level][model_name] = model_scores
    return model_scores_dict



def calc_score_and_std_per_timestep(X_test_df, y_test, y_pred, aggregated_timestamp_column='time_step', metric= 'f1'):

    last_train_time_step = min(X_test_df['time_step']) - 1
    last_time_step = max(X_test_df['time_step'])
    model_scores = []
    for time_step in range(last_train_time_step + 1, last_time_step + 1):
        time_step_idx = np.flatnonzero(X_test_df[aggregated_timestamp_column] == time_step)
        y_true_ts = y_test.iloc[time_step_idx]
        y_pred_ts = [y_pred[i] for i in time_step_idx]
        model_scores.append(calculate_model_score(y_true_ts.astype('int'), y_pred_ts, metric))

    f1_timestep = np.array(model_scores)
    return f1_timestep