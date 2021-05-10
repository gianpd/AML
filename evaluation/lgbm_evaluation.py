from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

import mlflow

from utils import run_elliptic_preprocessing_pipeline, get_unbalanced_weights
from mlflow_utils import log_binary_mlflow

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

EXPERIMENT_NAME = 'LGB'
PLOTS_ROOT = '../plots/plots_0307'

X_train_df, X_test_df, y_train_all, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

folds = TimeSeriesSplit(test_size=int(len(X_train_df) * 0.1), n_splits=5)

X_train_all = X_train_df.values
X_test = X_test_df.values
params = {'n_estimators':      350,
          'num_leaves':        17,
          'min_child_samples': 29,
          'objective':         'binary',
          'learning_rate':     0.031188616474561084,
          'subsample':         1,
          'colsample_bytree':  0.3774966956988639,
          'reg_alpha':         0.0019504606411800377,
          'reg_lambda':        14.065658041501804
          }

run_params = {
    'experiment_name': EXPERIMENT_NAME,
    'artifact_dir': PLOTS_ROOT,
    'nonce': uuid4().hex[:7]
}

cols = X_train_df.columns.to_list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = cols

sns.set_theme(style='darkgrid')
avg_f1 = []
avg_micro_f1 = []
mlflow.set_experiment('Elliptic - with Tuning')
with mlflow.start_run(run_name='TimeSeriesSplits'):
    for fold_n, (train_idx, test_idx) in enumerate(folds.split(X_train_all)):
        print(f'FOLD: {fold_n}')
        print(f'TRAIN: {train_idx}, TEST: {test_idx}')
        X_train, y_train = X_train_all[train_idx], y_train_all[train_idx]
        X_valid, y_valid = X_train_all[test_idx], y_train_all[test_idx]
        print(f'Train shape: {X_train.shape}')
        print(f'Val shape: {X_valid.shape}')
        assert X_train.shape[0] == y_train.shape[0]
        assert X_valid.shape[0] == y_valid.shape[0]
        fig, ax = plt.subplots()
        sns.countplot(y=y_train, label='train', color='slateblue')
        sns.countplot(y=y_valid, label='val', color='turquoise')
        ax.set_title('Train/validation labels count')
        plt.legend()
        mlflow.log_figure(fig, f'labels_count_fold_{fold_n}.png')
        mlflow.lightgbm.autolog()
        run_name = f'{EXPERIMENT_NAME}_fold_{fold_n}'
        with mlflow.start_run(run_name=run_name, nested=True):
            dtrain = lgb.Dataset(X_train, label=y_train) #weight=get_unbalanced_weights(y_train, 0.3, 0.7))
            dvalid = lgb.Dataset(X_valid, label=y_valid) #weight=get_unbalanced_weights(y_valid, 0.3, 0.7))
            res = {}
            clf = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], evals_result=res, early_stopping_rounds=50,
                            verbose_eval=100)
            feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
            fig, ax = plt.subplots(figsize=(16, 12))
            sns.barplot(data=feature_importances.sort_values(by=f'fold_{fold_n + 1}', ascending=False).head(30),
                        x=f'fold_{fold_n + 1}', y='feature')
            ax.set_title('30 TOP feature importance')
            mlflow.log_figure(fig, f'{run_name}_feature_importance.png')
            # WIP: evaluation. For now get PR_curve and best TH for validation data and compute average results
            y_probas = clf.predict(X_valid, num_iteration=clf.best_iteration)
            f1, micro_f1 = log_binary_mlflow(run_params=run_params, y_true=y_valid, y_probas=y_probas)
            avg_f1.append(f1)
            avg_micro_f1.append(micro_f1)
        plt.close('all')
    mlflow.log_metric('avg_f1_val', np.mean(avg_f1))
    mlflow.log_metric('avg_microF1_val', np.mean(avg_micro_f1))

    # Evaluation
    dtrain = lgb.Dataset(X_train_all, label=y_train_all)
    clf = lgb.train(params, dtrain)
    y_probas = clf.predict(X_test, num_iteration=clf.best_iteration)
    run_params['nonce'] = 'EVALUATION'
    f1, micro_f1 = log_binary_mlflow(run_params=run_params, y_true=y_test, y_probas=y_probas)



# plot average feature importance over k-folds
# feature_importances['average'] = feature_importances[
#     [f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
# feature_importances.to_csv('feature_importances.csv')
# plt.figure(figsize=(16, 12))
# sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature')
# plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits))
