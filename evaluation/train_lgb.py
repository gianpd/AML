from sklearn.metrics import f1_score

import mlflow

from train.train_lgb_utils import *
from utils import split_train_val_eval
from evaluation.plot_evaluation import plot_precision_recall_roc


SEED = 345
# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_eval(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

best_params = {
    'nthread':           6,
    'objective':         'binary',
    'n_estimators':      350,
    'num_leaves':        17,
    'min_child_samples': 29,
    'learning_rate':     0.031188616474561084,
    'subsample':         1.0,
    'log_max_bin':       5.0,
    'colsample_bytree':  0.3774966956988639,
    'reg_alpha':         0.0019504606411800377,
    'reg_lambda':        14.065658041501804,
    'seed':              SEED}

best_params_f1 = {
    'nthread':           4,
    'objective':         'xentropy',
    'metric':            'xentropy',
    'n_estimators':      350,
    'num_leaves':        12,
    'min_child_samples': 42,
    'learning_rate':     0.10343098206855576,
    'subsample':         0.6742865653847605,
    'log_max_bin':       4.0,
    'colsample_bytree':  0.39175448075941094,
    'reg_alpha':         0.0009765625,
    'reg_lambda':        0.025606287948153804,
    'seed':              SEED}

best = {
    'nthread':           4,
    'objective':         'binary',
    'n_estimators':      1400,
    'num_leaves':        8,
    'min_child_samples': 15,
    'learning_rate':     0.12780655980520544,
    'subsample':         0.8840015302357942,
    'log_max_bin':       7.0,
    'colsample_bytree':  1.0,
    'reg_alpha':         0.009012949417347304,
    'reg_lambda':        0.13074621698713804}

mlflow.lightgbm.autolog()
mlflow.set_experiment('LightGBM Optimal params')
with mlflow.start_run(run_name='best_params_f1') as run:
    model, _ = train_model(X_train, X_val, y_train, y_val, best)
    y_probs = model.predict(X_test)
    best_f1, best_th = plot_precision_recall_roc(y_test, y_probs, path='lgb_best_f1')
    y_pred = np.where(y_probs >= best_th, 1, 0)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric('f1', f1)
