from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

import mlflow

from utils import run_elliptic_preprocessing_pipeline
from mlflow_utils import log_binary_mlflow

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

EXPERIMENT_NAME = 'RF'
PLOTS_ROOT = '../plots/plots_0307'

X_train_df, X_test_df, y_train_all, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

folds = TimeSeriesSplit(test_size=int(len(X_train_df) * 0.1), n_splits=5)

X_train_all = X_train_df.values
X_test = X_test_df.values
run_params = {
    'experiment_name': EXPERIMENT_NAME,
    'artifact_dir':    PLOTS_ROOT,
    'nonce':           uuid4().hex[:7]
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
        mlflow.sklearn.autolog()
        run_name = f'{EXPERIMENT_NAME}_fold_{fold_n}'
        with mlflow.start_run(run_name=run_name, nested=True):
            model = RandomForestClassifier(n_estimators=150,
                                           #class_weight={0: 0.3, 1: 0.7},
                                           max_features=0.8730950943488909,
                                           criterion='entropy')
            clf = model.fit(X_train, y_train)
            # WIP: evaluation. For now get PR_curve and best TH for validation data and compute average results
            y_probas = clf.predict(X_valid)
            f1, micro_f1 = log_binary_mlflow(run_params=run_params, y_true=y_valid, y_probas=y_probas)
            avg_f1.append(f1)
            avg_micro_f1.append(micro_f1)
        plt.close('all')
    mlflow.log_metric('avg_f1_val', np.mean(avg_f1))
    mlflow.log_metric('avg_microF1_val', np.mean(avg_micro_f1))

    # Evaluation
    clf = model.fit(X_train_all, y_train_all)
    y_probas = clf.predict(X_test)
    run_params['nonce'] = 'EVALUATION'
    f1, micro_f1 = log_binary_mlflow(run_params=run_params, y_true=y_test, y_probas=y_probas)
