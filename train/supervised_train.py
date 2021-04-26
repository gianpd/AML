import os
import sys

from time import time
from typing import Optional, Union, List, Callable

# supervised baseline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.class_weight import compute_class_weight

# feature/model selection
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_validate

import mlflow

from evaluation.model_performance import *

import logging
logging.basicConfig(stream=sys.stdout, format='',
                level=logging.INFO, datefmt=None)
logger = logging.getLogger('supervised_train')

class Supervised:

    def __init__(self, model: str, task: str,
                 X_train,
                 y_train,
                 config: Optional[dict] = None,
                 X_val=None,
                 y_val=None,
                 num_cv=5,
                 class_weight=None,
                 seed=123):

        np.random.seed(seed)
        self._model = model
        self._task = task
        self._config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self._clf = None
        self._cv = num_cv
        self._class_weight = class_weight


    def _validate_data(self):
        pass

    def _preprocess(self, X):
        return X

    def _train_cv(self):
        log_index = '_train_cv>'
        scoring = ['f1', 'f1_micro']

        if self._class_weight:
            weight = self.get_balanced_weights(self.y_train)
            score = cross_validate(self._clf, self.X_train, self.y_train,
                                   cv=self._cv, scoring=scoring, fit_params={'sample_weight': weight})
        else:
            score = cross_validate(self._clf, self.X_train, self.y_train,
                                   cv=self._cv, scoring=scoring)

        avg_f1_test = np.mean(score["test_f1"])
        avg_f1_micro_test = np.mean(score["test_f1_micro"])
        logger.info(f'avg test_f1: {avg_f1_test}')
        logger.info(f'avg test_f1_micro: {avg_f1_micro_test}')
        if mlflow.active_run():
            logger.info(f'{log_index} - starting mlflow run ...')
            with mlflow.start_run(nested=True, run_name=f'{self._model}_{self._cv}_{self._class_weight}_train') as child_run:
                logger.info(f'run id: {child_run.info.run_id}')
                mlflow.log_metrics(
                    {
                        "{}_test_avgF1".format(self._model):      avg_f1_test,
                        "{}_test_avgF1Micro".format(self._model): avg_f1_micro_test,
                    }
                )
                mlflow.log_params(score)
                mlflow.log_params({'model':        self._model,
                                   'cv':           self._cv,
                                   'class_weight': self._class_weight,
                                   })

    def train_cv(self):

        start = time()

        if self._model == 'rf':
            if self._task in ('binary', 'multiclass'):
                self._clf = RandomForestClassifier()
            else:
                raise ValueError('regression task not yet implemented.')
        elif self._model == 'lgbm':
            if self._task in ('binary', 'multiclass'):
                self._clf = LGBMClassifier()
            else:
                raise ValueError('regression task not yet implemented.')
        elif self._model == 'lr':
            if self._task in ('binary', 'multiclass'):
                self._clf = LogisticRegression(max_iter=10000)
            else:
                raise ValueError('regression task not yet implemented.')
        elif self._model == 'xgboost':
            if self._task in ('binary', 'multiclass'):
                self._clf = XGBClassifier()
            else:
                raise ValueError('regression task not yet implemented.')
        else:
            raise ValueError(f'Classifier {self._model} not available.')

        self._train_cv()
        elapsed = time() - start
        logger.info(f'{self._model} train cv elapsed time: {elapsed} [s]')


    def predict(self, X_test):
        '''Predict label from features

        Args:
            X_test: A numpy array of featurized instances, shape n*m

        Returns:
            A numpy array of shape n*1.
            Each element is the label for a instance
        '''
        if self._clf is not None:
            X_test = self._preprocess(X_test)
            return self._clf.predict(X_test)
        else:
            return np.ones(X_test.shape[0])

    def predict_proba(self, X_test):
        '''Predict the probability of each class from features

        Only works for classification problems

        Args:
            model: An object of trained model with method predict_proba()
            X_test: A numpy array of featurized instances, shape n*m

        Returns:
            A numpy array of shape n*c. c is the # classes
            Each element at (i,j) is the probability for instance i to be in
                class j
        '''
        if 'regression' in self._task:
            raise ValueError('Regression tasks do not support predict_prob')
        else:
            X_test = self._preprocess(X_test)
            return self._clf.predict_proba(X_test)

    def evaluate(self):
        log_index = 'evaluate>'
        if self.X_val is None and self.y_val is None:
            raise ValueError('Evaluation dataset not provided.')
        if hasattr(self._clf, 'predict'):
            self._clf.fit(self.X_train, self.y_train)
            y_pred = self._clf.predict(self.X_val)
            model_scores = calculate_model_score(self.y_val, y_pred)
            if mlflow.active_run():
                with mlflow.start_run(nested=True, run_name=f'{self._model}_{self._cv}_{self._class_weight}_eval') as child_run:
                    logger.info(f'{log_index} - starting mlflow Tracking ...')
                    mlflow.log_params(
                        {
                            "{}_eval_accuracy".format(self._model):  model_scores['accuracy'],
                            "{}_eval_f1".format(self._model):        model_scores['f1'],
                            "{}_eval_microF1".format(self._model):   model_scores['f1_micro'],
                            "{}_eval_macroF1".format(self._model):   model_scores['f1_macro'],
                            "{}_eval_precision".format(self._model): model_scores['precision'],
                            "{}_eval_recall".format(self._model):    model_scores['recall'],
                            "{}_eval_rocAUC".format(self._model):    model_scores['roc_auc']
                        })
                    params = self._clf.get_params()
                    for k, v in params.items():
                        mlflow.log_params(
                            {
                                "{}_{}".format(self._model, k): v
                            }
                        )
            else:
                logger.info('no active mlflow run.')
                logger.info(f'{self._model} - scores: {model_scores}')

            return y_pred

        else:
            raise ValueError('classifier not provided.')

    def get_balanced_weights(self, labels):
        class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
        return weights
