import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
import matplotlib.lines as lines
current_working_directory = os.getcwd()
SAVEDIR = os.path.dirname(current_working_directory)


def calculate_rmse(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    mse = np.mean((true_labels - pred_labels) ** 2)  #
    rmse = np.sqrt(mse)  #
    return rmse
#%%
def exact_accuracy(true_labels, pred_labels):
    success_count = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            success_count += 1
    accuracy = success_count / len(true_labels)
    return accuracy
#%%
def adjacency_accuracy(true_labels, pred_labels):
    success_count = 0
    for true, pred in zip(true_labels, pred_labels):
        if abs(true - pred) <= 1:
            success_count += 1
    accuracy = success_count / len(true_labels)
    return accuracy


rmse_matrix_dict = dict()
adjacc_matrix_dict = dict()
exacc_matrix_dict = dict()
