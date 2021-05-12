import mlflow
from flaml import AutoML
from flaml.data import get_output_from_log

from utils import split_train_val_eval
from model_performance import *
from plot_evaluation import *


# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

TIME_BUDGET = 1800
PLOTS_ROOT = '../plots'

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP,
                                                                               LAST_TIMESTEP)

X_train = X_train_df.values
X_test = X_test_df.values

#AUTO TUNING
mlflow.set_experiment(f'AutoML Tuning - Elliptic')
with mlflow.start_run() as run:
    for m in ['rf']:
        automl = AutoML()
        settings = {
            "time_budget":         TIME_BUDGET,
            "task":                'binary',
            "estimator_list":      [m],
            "log_file_name":       f'automl_{m}.log',
            "eval_method":        'cv',
            "n_splits":            10,
            "log_training_metric": True,
            "model_history":       True,
            "verbose":             1
        }
        mlflow.lightgbm.autolog() if m == 'lgbm' else mlflow.sklearn.autolog()
        with mlflow.start_run(nested=True, run_name=f'Automl-{m}'):
            mlflow.log_param('TIME_BUDGET', TIME_BUDGET)
            mlflow.log_params(settings)

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
            mlflow.log_figure(fig, f'f1_timestep_{m}_automl.png')

            plot_confusion_matrix(y_test, y_pred, path=f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
            plot_precision_recall_roc(y_test, y_prob, path=f'{PLOTS_ROOT}/automl_{m}')
            mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_confusion_matrix.png')
            mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_precision_recall.png')
            mlflow.log_artifact(f'{PLOTS_ROOT}/automl_{m}_roc.png')

            plt.close('all')
