from train.supervised_train import Supervised

from utils import run_elliptic_preprocessing_pipeline
from plot_evaluation import *
from model_performance import *

import mlflow

import matplotlib.pyplot as plt
import numpy as np

PLOTS_ROOT = '../plots/plots_0307'

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

SEED = 456
CVs = [15]
CLASS_WEIGHTS = [False]
CLFs = ['lgbm', 'rf']

# auto ml parameters
TIME_BUDGET = 60 * 15

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
            c_weight = '0307' if class_weight else 0
            for m in CLFs:
                mlflow.sklearn.autolog()
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
                        f'f1':        model_scores['f1'],
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


