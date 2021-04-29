from train.supervised_train import Supervised

from utils import *
from plot_evaluation import *
from model_performance import *

from flaml import AutoML
from flaml.data import get_output_from_log

import mlflow

import matplotlib.pyplot as plt
import numpy as np

PLOTS_ROOT = '../plots'

# 70:20
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

SEED = 456
CVs = [5, 10, 15]
CLASS_WEIGHTS = [False]
CLFs = ['lgbm', 'rf']

# auto ml parameters
TIME_BUDGET = 60 * 30

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

X_train = X_train_df.values
X_test = X_test_df.values

mlflow.set_experiment(f'Elliptic - with Tuning')
with mlflow.start_run():
    mlflow.log_params({
        'LAST_TRAIN_TIMESTEP': LAST_TRAIN_TIMESTEP,
        'LAST_TIMESTEP':       LAST_TIMESTEP,
        'SEED':                SEED,
        'CVs':                 CVs,
        'CLASS_WEIGHTS':       CLASS_WEIGHTS,
        'CLFs':                CLFs
    })
    for cv in CVs:
        for class_weight in CLASS_WEIGHTS:
            c_weight = 1 if class_weight else 0
            for m in CLFs:
                with mlflow.start_run(nested=True, run_name=f'{m}-{cv}-{c_weight}') as run:
                    print('### START')
                    print(f'cv: {cv} - class_weight: {c_weight} - model: {m}')
                    mlflow.set_tags({'model':        m,
                                     'cv':           cv,
                                     'class_weight': class_weight,
                                     })
                    clf = Supervised(
                        model=m,
                        task='binary',
                        X_train=X_train,
                        y_train=y_train,
                        num_cv=cv,
                        class_weight=class_weight,
                        seed=SEED)
                    score = clf.train_cv()
                    avg_f1_test = np.mean(score['test_f1'])
                    std_f1_test = np.std(score['test_f1'])
                    avg_micro_f1_test = np.mean(score['test_f1_micro'])
                    std_micro_f1_test = np.std(score['test_f1_micro'])
                    mlflow.log_metrics({
                        f'avg_f1_test':       avg_f1_test,
                        f'std_f1_test':       std_f1_test,
                        f'avg_micro_f1_test': avg_micro_f1_test,
                        f'std_micro_f1_test': std_micro_f1_test,
                    })

                    y_pred = clf.evaluate(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    model_scores = calculate_model_score(y_test, y_pred)
                    mlflow.log_metrics({
                        f'accuracy':  model_scores['accuracy'],
                        f'f1': model_scores['f1'],
                        f'f1_micro':  model_scores['f1_micro'],
                        f'f1_macro':  model_scores['f1_macro'],
                        f'precision': model_scores['precision'],
                        f'recall':    model_scores['recall'],
                        f'roc_auc':   model_scores['roc_auc']
                    })

                    f1_timestep = calc_score_and_std_per_timestep(X_test_df, y_test, y_pred)
                    fig, ax = plt.subplots()
                    ax.plot(range(LAST_TRAIN_TIMESTEP + 1, LAST_TIMESTEP + 1), f1_timestep)
                    ax.set_xlabel('timestep')
                    ax.set_ylabel('f1')
                    mlflow.log_figure(fig, f'f1_timestep_{m}.png')

                    plot_confusion_matrix(y_test, y_pred, path=f'{PLOTS_ROOT}/{m}_{cv}_{c_weight}_confusion_matrix.png')
                    plot_precision_recall_roc(y_test, y_prob, path=f'{PLOTS_ROOT}/{m}_{cv}_{c_weight}')
                    mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_{cv}_{c_weight}_confusion_matrix.png')
                    mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_{cv}_{c_weight}_precision_recall.png')
                    mlflow.log_artifact(f'{PLOTS_ROOT}/{m}_{cv}_{c_weight}_roc.png')

plt.close('all')

# # AUTO TUNING
# mlflow.set_experiment(f'AutoML Tuning - Elliptic')
# with mlflow.start_run() as run:
#     for m in ['lgbm', 'rf']:
#         automl = AutoML()
#         settings = {
#             "time_budget":         TIME_BUDGET,
#             "metric":              'micro_f1',
#             "task":                'binary',
#             "estimator_list":      [m],
#             "log_file_name":       f'automl_{m}.log',
#             "log_training_metric": True,
#             "model_history":       True,
#             "verbose":             1
#         }
#         with mlflow.start_run(nested=True, run_name=f'Automl-{m}'):
#             mlflow.log_param('TIME_BUDGET', TIME_BUDGET)
#             mlflow.log_params(settings)
#
#             automl.fit(X_train=X_train,
#                        y_train=y_train,
#                        X_val=X_val,
#                        y_val=y_val,
#                        **settings)
#             print('### AUTO ML')
#             print('Best hyperparmeter config:', automl.best_config)
#             print('Best log_loss on validation data: {0:.4g}'.format(automl.best_loss))
#             print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
#             time_history, best_valid_loss_history, valid_loss_history, config_history, train_loss_history = \
#                 get_output_from_log(filename=settings['log_file_name'], time_budget=TIME_BUDGET)
#             plt.title(f'Learning Curve - {m}')
#             plt.xlabel('Wall Clock Time (s)')
#             plt.ylabel('Validation Accuracy')
#             plt.scatter(time_history, 1 - np.array(valid_loss_history))
#             plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
#             file_path = f'{PLOTS_ROOT}/automl_learning_curve_{m}.png'
#             plt.savefig(file_path)
#             plt.close()
#             mlflow.log_artifact(file_path)
#
#             y_pred = automl.predict(X_test)
#             y_prob = automl.predict_proba(X_test)[:, 1]
#
#             model_scores = calculate_model_score(y_test, y_pred)
#             mlflow.log_metrics({
#                 f'accuracy':  model_scores['accuracy'],
#                 f'f1': model_scores['f1'],
#                 f'f1_micro':  model_scores['f1_micro'],
#                 f'f1_macro':  model_scores['f1_macro'],
#                 f'precision': model_scores['precision'],
#                 f'recall':    model_scores['recall'],
#                 f'roc_auc':   model_scores['roc_auc']
#             })
#
#             f1_timestep = calc_score_and_std_per_timestep(X_test, y_test, y_pred)
#             fig, ax = plt.subplots()
#             ax.plot(range(LAST_TRAIN_TIMESTEP + 1, LAST_TIMESTEP + 1), f1_timestep)
#             ax.set_xlabel('timestep')
#             ax.set_ylabel('f1')
#             mlflow.log_figure(fig, f'f1_timestep_{m}.png')
#
#             plot_confusion_matrix(y_test, y_pred, path=f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
#             plot_precision_recall_roc(y_test, y_prob, path=f'{PLOTS_ROOT}/automl_{m}')
#             mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
#             mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_precision_recall.png')
#             mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_roc.png')
#
#             plt.close('all')