from train.supervised_train import Supervised

from utils import *

from model_performance import *

from flaml import AutoML

import mlflow

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

clf_list = ['lgbm', 'rf', 'lr', 'xgboost']

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

X_train = X_train_df.values
X_test = X_test_df.values

SEED = 456

for m in clf_list:
    clf = Supervised(model=m,
                     task='binary',
                     X_train=X_train,
                     y_train=y_train,
                     X_val=X_test,
                     y_val=y_test,
                     seed=SEED)
    clf.train_cv()
    clf.evaluate()



# automl = AutoML()
#
# settings = {
#     "time_budget":    60,
#     "metric":         'log_loss',
#     "estimator_list": ['lgbm', 'rf'],
#     "task":           'binary',
#     "log_file_name":  'automl.log',
# }

# mlflow.sklearn.autolog()
# mlflow.set_experiment('Elliptic AutoML')
# with mlflow.start_run() as run:
#     automl.fit(X_train=X_train,
#                y_train=y_train,
#                X_val=X_test,
#                y_val=y_test,
#                **settings)
#
# print('### AUTO ML')
# print('Best classifier:', automl._best_estimator)
# print('Best hyperparmeter config:', automl.best_config)
# print('Best log_loss on validation data: {0:.4g}'.format(automl.best_loss))
# print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
