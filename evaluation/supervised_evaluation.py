from train.supervised_train import Supervised

from utils import *
from plot_evaluation import *

from flaml import AutoML
from flaml.data import get_output_from_log

import mlflow

import matplotlib.pyplot as plt
import numpy as np

PLOTS_ROOT = '../plots'

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

SEED = 456
CVs = [5, 10, 15]
CLASS_WEIGHTS = [False, True]
CLFs = ['lgbm', 'rf', 'lr', 'xgboost']

# auto ml parameters
TIME_BUDGET = 1800

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)
X_train = X_train_df.values
X_test = X_test_df.values

mlflow.set_experiment(f'Elliptic')
with mlflow.start_run() as run:
    for cv in CVs:
        for class_weight in CLASS_WEIGHTS:
            for m in CLFs:
                print('### START')
                print(f'cv: {cv} - class_weight: {class_weight} - model: {m}')
                clf = Supervised(
                    model=m,
                    task='binary',
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_test,
                    y_val=y_test,
                    num_cv=cv,
                    class_weight=class_weight,
                    seed=SEED
                )
                clf.train_cv()
                y_pred = clf.evaluate()
                y_prob = clf.predict_proba(X_test)[:, 1]
                plot_confusion_matrix(y_test, y_pred, path=f'{PLOTS_ROOT}/{m}_confusion_matrix.png')
                plot_precision_recall_roc(y_test, y_prob, path=f'{PLOTS_ROOT}/{m}')
                mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_confusion_matrix.png')
                mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_precision_recall.png')
                mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_roc.png')



# AUTO TUNING
for m in ['lgbm', 'rf']:
    automl = AutoML()
    settings = {
        "time_budget":    TIME_BUDGET,
        "metric":         'log_loss',
        "task":           'binary',
        "estimator_list": [m],
        "log_file_name":  f'automl_{m}.log'
    }
    mlflow.set_experiment(f'AutoML Tuning - Elliptic {m}')
    with mlflow.start_run() as run:
        automl.fit(X_train=X_train,
                   y_train=y_train,
                   X_val=X_test,
                   y_val=y_test,
                   **settings)
        print('### AUTO ML')
        print('Best hyperparmeter config:', automl.best_config)
        print('Best log_loss on validation data: {0:.4g}'.format(automl.best_loss))
        print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
        time_history, best_valid_loss_history, valid_loss_history, config_history, train_loss_history = \
            get_output_from_log(filename=settings['log_file_name'], time_budget=TIME_BUDGET)
        plt.title(f'Learning Curve - {m}')
        plt.xlabel('Wall Clock Time (s)')
        plt.ylabel('Validation Accuracy')
        plt.scatter(time_history, 1 - np.array(valid_loss_history))
        plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
        file_path = f'{PLOTS_ROOT}/automl_learning_curve_{m}.png'
        plt.savefig(file_path)
        plt.close()
        mlflow.log_artifact(file_path)

        y_pred = automl.predict(X_test)
        y_prob = automl.predict_proba(X_test)[:, 1]
        plot_confusion_matrix(y_test, y_pred, path=f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
        plot_precision_recall_roc(y_test, y_prob, path=f'{PLOTS_ROOT}/automl_{m}')
        mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
        mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_precision_recall.png')
        mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_roc.png')
