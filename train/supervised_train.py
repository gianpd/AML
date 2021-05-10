import os
import sys

from time import time
from typing import Optional, Union, List, Callable

# supervised baseline
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.class_weight import compute_class_weight

# feature/model selection
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, FeatureUnion
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
            weight = self.get_unbalanced_weights(self.y_train, 0.3, 0.7)
            #self._clf = make_pipeline(PCA(n_components=85), self._clf)
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
                self._clf = RandomForestClassifier(n_estimators=150,
                                                   #class_weight={0: 0.3, 1: 0.7},
                                                   max_features=0.8730950943488909,
                                                   criterion='entropy')
            else:
                raise ValueError('regression task not yet implemented.')
        elif self._model == 'lgbm':
            if self._task in ('binary', 'multiclass'):
                self._clf = LGBMClassifier(n_estimators=350,
                                           num_leaves=17,
                                           class_weight={0: 0.3, 1: 0.7},
                                           min_child_samples=29,
                                           objective='binary',
                                           learning_rate=0.031188616474561084,
                                           subsample=1,
                                           colsample_bytree=0.3774966956988639,
                                           reg_alpha=0.0019504606411800377,
                                           reg_lambda=14.065658041501804)
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

    def evaluate(self, X_test):
        log_index = 'evaluate>'
        if hasattr(self._clf, 'predict'):
            #self._clf = make_pipeline(PCA(n_components=75), self._clf)
            self._clf.fit(self.X_train, self.y_train)
            y_pred = self._clf.predict(X_test)
            return y_pred
        else:
            raise ValueError('classifier not provided.')

    def get_balanced_weights(self, labels):
        class_weight = compute_class_weight(classes=np.unique(labels), y=labels)
        weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
        return weights

    def get_unbalanced_weights(self, labels, weight1, weight2):
        class_weight = compute_class_weight({0: weight1, 1: weight2}, classes=np.unique(labels), y=labels)
        weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
        return weights

if __name__ == '__main__':

    from utils import run_elliptic_preprocessing_pipeline

    X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(34, 49)

    X_train = X_train_df.values
    X_test = X_test_df.values

    clf = Supervised(
        model='rf',
        task='binary',
        X_train=X_train,
        y_train=y_train,
        num_cv=5
    )

    w = clf.get_weights(y_train)
    print(w)
