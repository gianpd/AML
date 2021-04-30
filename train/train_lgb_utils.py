import sys

import time

from datetime import datetime

import numpy as np

import lightgbm as lgb

import logging

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

logging.basicConfig(stream=sys.stdout, format='', level=logging.INFO, datefmt=None)
logger = logging.getLogger('train')

ROOT_DIR = '../models'


def get_filename_prefix():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_weights(labels):
    class_weight = compute_class_weight({0: 0.3, 1: 0.7}, classes=np.unique(labels), y=labels)
    weights = labels.map(lambda x: class_weight[0] if not x else class_weight[1])
    return weights

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)
    return 'f1', f1_score(y_true, y_hat), True

def train_model(X_train_df, X_val_df, y_train, y_val, parameters):

    linear_tree = parameters.get('linear_tree')

    train_weight = get_weights(y_train)

    t = time.time()

    last_time_step = min(X_val_df['time_step'])
    last_train_time_step = max(X_train_df['time_step'])

    logger.info(
        f'Training model: LAST_UPDATE from {last_train_time_step} to {last_time_step}')
    logger.info(f'parameters {parameters}')

    train_data = lgb.Dataset(X_train_df, label=y_train, weight=train_weight)

    val_data = lgb.Dataset(X_val_df, label=y_val)

    valid_sets = [train_data, val_data]
    valid_names = ['train', 'valid']

    res = {}
    t0 = time.time()
    time_arr_linear = []
    timer_callback = lambda env: time_arr_linear.append(time.time() - t0)
    model = lgb.train(parameters, train_data, valid_sets=valid_sets, valid_names=valid_names,
                      evals_result=res, callbacks=[timer_callback], feval=lgb_f1_score)
    linear_time = np.round(time.time() - t, 2)
    title = 'Linear Tree model' if linear_tree else 'Constant model'
    logger.info(title)
    logger.info("training time: {} [s]".format(linear_time))
    return model, res
