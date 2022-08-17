import os

from IPython.display import display, Markdown

import numpy as np

# evaluation
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support

import scikitplot as skplt
import matplotlib.pyplot as plt

from AML.utils import calc_occurences_per_timestep
from ..utils import load_elliptic_data


ROOT_DIR = os.path.join(os.pardir, __file__)


def plot_precision_recall_roc(y_test, y_prob, path=None):

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * recall * precision / (recall + precision)
    # TODO: by removing nan f1-scores thresholds indexes are no more aligned
    f1_scores = f1_scores[~np.isnan(f1_scores)]

    best_threshold = thresholds[np.argmax(f1_scores)]
    display(Markdown(f'Best threshold: {best_threshold:.3f}'))
    best_f1 = np.max(f1_scores)
    display(Markdown(f'Best F1-Score: {best_f1:.3f}'))

    average_precision = average_precision_score(y_test, y_prob)
    display(Markdown(f'Average precision: {average_precision:.3f}'))
    display(Markdown(f'Percentage of true labels: {y_test.sum() / len(y_test):.3f}'))

    probas = np.column_stack((1 - y_prob, y_prob))
    skplt.metrics.plot_precision_recall(y_test, probas)
    if path is not None:
        plt.savefig(f'{path}_precision_recall.png')
        skplt.metrics.plot_roc(y_test, probas)
        plt.savefig(f'{path}_roc.png')
        plt.close()
    else:
        plt.show()
        skplt.metrics.plot_roc(y_test, probas)
        plt.show()

    return best_f1, best_threshold

def plot_confusion_matrix(y_true, y_pred, path=None, title=None, xtickslabels=None, ytickslabels=None):

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

    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_performance_per_timestep(root_dataset_path, model_metric_dict, last_train_time_step=34, last_time_step=49, model_std_dict=None,
                                  fontsize=23, labelsize=18, figsize=(20, 10),
                                  markers=['^', '<', 'p', 'o'], 
                                  linestyles=['f', 'f', 'f', 'f'],
                                  linecolor=["green", "orange", "red", 'blue'],
                                  barcolor='lightgrey', baralpha=0.3, linewidth=1.5, savefig_path=None):

    X, y = load_elliptic_data(root_dataset_path=root_dataset_path)
    occ = calc_occurences_per_timestep(X, y)
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]
    plt.style.use('grayscale')

    timesteps = illicit_per_timestep['time_step'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    num_curves = len(list(model_metric_dict.keys()))
    markers = markers if len(markers) ==  num_curves else markers[:num_curves]
    i = 0
    for key, values in model_metric_dict.items():
        if key != "XGBoost":
            key = key.lower()
        # values = values[0].flatten()
        ax1.plot(timesteps, values, label=key, linestyle=linestyles[i], marker=markers[i], color=linecolor[i], linewidth=linewidth)
        if model_std_dict != None:
            ax1.fill_between(timesteps, values + model_std_dict[key], values - model_std_dict[key],
                             facecolor='lightgrey', alpha=0.5)
        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha, label='\# illicit')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Time step', fontsize=fontsize)
    ax1.set_ylabel('Illicit F1', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step + 1, last_time_step + 1))
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Num. samples', fontsize=fontsize)

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(os.path.join(ROOT_DIR, savefig_path), bbox_inches='tight', pad_inches=0)
