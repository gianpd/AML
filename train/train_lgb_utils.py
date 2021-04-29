import sys

import time

from datetime import datetime

import numpy as np

import lightgbm as lgb

import logging

logging.basicConfig(stream=sys.stdout, format='', level=logging.INFO, datefmt=None)
logger = logging.getLogger('train')

ROOT_DIR = '../models'


def get_filename_prefix():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def train_model(X_train_df, X_val_df, y_train, y_val, parameters, label=None):

    linear_tree = parameters.get('linear_tree')

    t = time.time()

    last_time_step = min(X_val_df['time_step'])
    last_train_time_step = max(X_train_df['time_step'])

    logger.info(
        f'Training model: LAST_UPDATE from {last_train_time_step} to {last_time_step}')
    logger.info(f'parameters {parameters}')

    train_data = lgb.Dataset(X_train_df, label=y_train)

    val_data = lgb.Dataset(X_val_df, label=y_val)

    valid_sets = [train_data, val_data]
    valid_names = ['train', 'valid']

    res = {}
    t0 = time.time()
    time_arr_linear = []
    timer_callback = lambda env: time_arr_linear.append(time.time() - t0)
    model = lgb.train(parameters, train_data, valid_sets=valid_sets, valid_names=valid_names,
                      evals_result=res, callbacks=[timer_callback])
    linear_time = np.round(time.time() - t, 2)
    title = 'Linear Tree model' if linear_tree else 'Constant model'
    logger.info(title)
    logger.info("training time: {} [s]".format(linear_time))
    return model, res
