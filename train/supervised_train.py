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
        return score

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

        score = self._train_cv()
        elapsed = time() - start
        logger.info(f'{self._model} train cv elapsed time: {elapsed} [s]')
        return score

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
            return y_pred
        else:
            raise ValueError('classifier not provided.')

    def get_balanced_weights(self, labels):
        class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
        return weights
