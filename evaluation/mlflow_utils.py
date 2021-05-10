import os
import sys

import json

from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import scikitplot as skplt

import mlflow

from IPython.display import display, Markdown


def evaluate_multiclass(y_true, y_probas):
    loss = log_loss(y_true, y_probas)
    class_report = classification_report(y_true, np.argmax(y_probas, axis=1), output_dict=True)

    return loss, class_report


def evaluate_binary(y_true, y_probas, threshold):
    y_pred_threshold = np.where(y_probas >= threshold, 1, 0)

    auc = roc_auc_score(y_true, y_probas)
    recall = recall_score(y_true, y_pred_threshold)
    precision = precision_score(y_true, y_pred_threshold)
    f1 = f1_score(y_true, y_pred_threshold)
    micro_f1 = f1_score(y_true, y_pred_threshold, average='micro')

    return auc, recall, precision, f1, micro_f1


def log_binary_mlflow(run_params, y_true, y_probas):
    # mlflow.set_experiment(run_params['experiment_name'])

    pr_path, roc_path, avg_precision, best_th, best_f1_score = plot_precision_recall_roc(y_true,
                                                                                         y_probas,
                                                                                         plot_path_dir=run_params[
                                                                                             'artifact_dir'])

    auc, recall, precision, f1, micro_f1 = evaluate_binary(y_true, y_probas, best_th)

    y_pred = np.where(y_probas >= best_th, 1, 0)
    confusion_matrix_path = plot_confusion_matrix(y_true, y_pred, plot_path_dir=run_params['artifact_dir'])

    with mlflow.start_run(nested=True, run_name=f"evaluation-{run_params['nonce']}"):
        mlflow.log_param('best_th', best_th)
        mlflow.log_metric('auc', auc)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('micro_f1', micro_f1)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(confusion_matrix_path)

    return f1, micro_f1


def log_multiclass_mlflow(run_params, y_true, y_probas):
    # mlflow.set_experiment(run_params['experiment_name'])

    loss, class_report = evaluate_multiclass(y_true, y_probas)

    class_report_path = f"./plots/{run_params['model_name']}_classification_report.json"
    with open(class_report_path, "+w") as f:
        json.dump(class_report, f)

    with mlflow.start_run(nested=True, run_name=f"evaluation-{run_params['nonce']}"):
        # mlflow.log_artifact(run_params['model'])
        mlflow.log_metric('log_loss', loss)
        mlflow.log_artifact(class_report_path)


# PLOTTING
def plot_precision_recall_roc(y_true, y_probas, plot_path_dir=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

    f1_scores = 2 * recall * precision / (recall + precision)
    # TODO: by removing nan f1-scores thresholds indexes are no more aligned
    f1_scores = f1_scores[~np.isnan(f1_scores)]

    best_threshold = thresholds[np.argmax(f1_scores)]
    display(Markdown(f'Best threshold: {best_threshold:.3f}'))
    best_f1_score = np.max(f1_scores)
    display(Markdown(f'Best F1-Score: {best_f1_score:.3f}'))

    average_precision = average_precision_score(y_true, y_probas)
    display(Markdown(f'Average precision: {average_precision:.3f}'))
    display(Markdown(f'Percentage of true labels: {y_true.sum() / len(y_true):.3f}'))

    probas = np.column_stack((1 - y_probas, y_probas))
    skplt.metrics.plot_precision_recall(y_true, probas)

    if plot_path_dir is not None:
        path_pr = f'{plot_path_dir}/precision_recall_curve.png'
        plt.savefig(path_pr)
        plt.close()

        path_roc = f'{plot_path_dir}/roc_curve.png'
        skplt.metrics.plot_roc(y_true, probas)
        plt.savefig(path_roc)
        return path_pr, path_roc, average_precision, best_threshold, best_f1_score
    else:
        plt.show()
        return average_precision, best_threshold, best_f1_score


def plot_confusion_matrix(y_true, y_pred, title=None, xtickslabels=None, ytickslabels=None, plot_path_dir=None):
    precision, recall, f1score, support = precision_recall_fscore_support(y_true, y_pred)

    display(Markdown(f'Precision {precision}, recall {recall}, f1score {f1score}, support {support}'))

    ax = skplt.metrics.plot_confusion_matrix(
        y_true,
        y_pred,
        normalize=True,
        figsize=(10, 8),
        title=title
    )

    if xtickslabels is not None:
        ax.set_xticklabels(xtickslabels)

    if ytickslabels is not None:
        ax.set_yticklabels(ytickslabels)

    if plot_path_dir is not None:
        path = f'{plot_path_dir}/confusion_matrix.png'
        plt.savefig(path)
        plt.close()
        return path
    else:
        plt.show()