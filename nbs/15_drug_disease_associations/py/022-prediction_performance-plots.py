# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.044577, "end_time": "2020-12-18T22:38:21.345879", "exception": false, "start_time": "2020-12-18T22:38:21.301302", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011764, "end_time": "2020-12-18T22:38:21.398073", "exception": false, "start_time": "2020-12-18T22:38:21.386309", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf
# from multiplier import MultiplierProjection
# from entity import Trait

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load PharmacotherapyDB gold standard

# %%
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %%
gold_standard['true_class'].value_counts()

# %%
gold_standard['true_class'].value_counts(normalize=True)

# %%
doids_in_gold_standard = set(gold_standard['trait'])

# %% [markdown]
# # Load PhenomeXcan data

# %%
# input_file = Path(
#     conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
#     "most_signif",
#     "spredixcan-most_signif.pkl"
# ).resolve()

# display(input_file)

# %%
# phenomexcan_data = pd.read_pickle(input_file)

# %%
# phenomexcan_data.shape

# %%
# phenomexcan_data = phenomexcan_data.dropna(how='any')

# %%
# phenomexcan_data.shape

# %%
# phenomexcan_data.head()

# %%
# assert phenomexcan_data.index.is_unique

# %% [markdown]
# # Load drug-disease predictions

# %%
from collections import defaultdict

# %%
current_prediction_files = list(OUTPUT_PREDICTIONS_DIR.glob("*.h5"))
display(len(current_prediction_files))

# %%
predictions = []

for f in current_prediction_files:
#     print(f.name)
    
    prediction_data = pd.read_hdf(f, key="prediction")
    prediction_data = pd.merge(
        prediction_data, gold_standard,
        on=['trait', 'drug'],
        how='inner'
    )
    
    metadata = pd.read_hdf(f, key="metadata")
    
#     new_predictions[f"{metadata.method}"][metadata.data] = prediction_data
    prediction_data['trait'] = prediction_data['trait'].astype('category')
    prediction_data['drug'] = prediction_data['drug'].astype('category')
    prediction_data = prediction_data.assign(method=metadata.method)
    prediction_data = prediction_data.assign(data=metadata.data)
    
    predictions.append(prediction_data)
    
#     print(f"  shape: {prediction_data.shape}")

# %%
predictions = pd.concat(predictions, ignore_index=True)

# %%
predictions.shape

# %%
predictions.head()


# %% [markdown]
# # Average predictions

# %%
def _reduce(x):
    return pd.Series({
        'score': x['score'].max(),
        'true_class': x['true_class'].unique()[0] if x['true_class'].unique().shape[0] == 1 else None,
        'data': x['method'].iloc[0],
    })


# %%
predictions_avg = predictions.groupby(['trait', 'drug', 'method']).apply(_reduce).dropna().sort_index().reset_index()

# %%
predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# # ROC

# %%
from sklearn.metrics import roc_auc_score

# %%
predictions.groupby(['method', 'data']).apply(lambda x: roc_auc_score(x['true_class'], x['score'])).groupby('method').describe()

# %%
predictions_avg.groupby(['method', 'data']).apply(lambda x: roc_auc_score(x['true_class'], x['score'])).groupby('method').describe()

# %% [markdown]
# # PR

# %%
from sklearn.metrics import average_precision_score

# %%
predictions.groupby(['method', 'data']).apply(lambda x: average_precision_score(x['true_class'], x['score'])).groupby('method').describe()

# %%
predictions_avg.groupby(['method', 'data']).apply(lambda x: average_precision_score(x['true_class'], x['score'])).groupby('method').describe()

# %% [markdown]
# # ROC curves

# %%
NOT FINISHED

# %%
methods_names = tuple(predictions.keys())
display(methods_names)

# %%
methods_colors = {
    'Module-based': 'red',
    'Gene-based': 'blue'
}

# %%
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# %%
def plot_roc(data, method_key, fig, ax, remove_non_informative=False):
    roc_auc = roc_auc_score(data['true_class'], data['score'])
    
    fpr, tpr, thresholds = roc_curve(data['true_class'], data['score'])
#     print(f'  Points for ROC curve: {len(fpr)}')
    if remove_non_informative:
        cond = (fpr < 1.0) & (tpr < 1.0)
        fpr = fpr[cond]
        tpr = tpr[cond]
        print(f'  Points for ROC curve (after prunning): {len(fpr)}')
    
    label = f'{method_key} - AUC: {roc_auc:.3f}'
    sns.lineplot(x=fpr, y=tpr, estimator=None, label=label, ax=ax, linewidth=0.75, linestyle="--", color=methods_colors[method_key])


# %%
def plot_roc_for_methods(selected_methods):
    with sns.plotting_context('paper', font_scale=3.00):
        fig, ax = plt.subplots(figsize=(10, 10))

        for k in selected_methods:
            for name, data in predictions[k].items():
                plot_roc(data, k, fig, ax)

#         ax.set_title('ROC curves using PharmacotherapyDB')
        ax.plot([0.0, 1.00], [0.0, 1.00], color='gray', linewidth=1.25, linestyle='-')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.01])
#         ax.legend(loc="lower right")
        ax.set_aspect('equal')
        ax.get_legend().remove()


# %%
plot_roc_for_methods(methods_names)
# plt.savefig('/tmp/roc.pdf', bbox_inches='tight')

# %% [markdown]
# # Precision-Recall curve

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import minmax_scale


# %%
def plot_pr_raw_data(recall, precision, label, **kwargs):
    sns.lineplot(x=recall, y=precision, label=label, **kwargs)

    
def plot_pr(data, method_key, fig, ax, estimator=None, remove_non_informative=False):
    precision, recall, thresholds = precision_recall_curve(data['true_class'], data['score'])
    if remove_non_informative:
        cond = (recall < 1.0)
        recall = recall[cond]
        precision = precision[cond]
    
    ap = average_precision_score(data['true_class'], data['score'])
    
    label = f'{method_key} - AP: {ap:.3f}'
    plot_pr_raw_data(recall, precision, label, estimator=estimator, ax=ax, linewidth=0.75, linestyle="--", color=methods_colors[method_key])


# %%
def get_random_classifier_pr(data, reps=10, min_val=0, max_val=1):
    random_precision = []
    random_recall = []
    random_average_precision = []

    for i in range(reps):
        random_score = np.random.permutation(data['score'].values)
        rp, rr, _ = precision_recall_curve(data['true_class'], random_score)
        random_precision.extend(rp)
        random_recall.extend(rr)
        random_average_precision.append(
            average_precision_score(data['true_class'], random_score)
        )
    
    return np.array(random_recall), np.array(random_precision), np.array(random_average_precision)


# %%
def plot_pr_for_methods(selected_methods):
    with sns.plotting_context('paper', font_scale=3.00):
        fig, ax = plt.subplots(figsize=(10, 10))

        for k in selected_methods:
            for name, data in predictions[k].items():
                plot_pr(data, k, fig, ax)

        # add random classifier
    #     random_recall, random_precision = get_random_classifier_pr(predictions['multixcan_mashr'].shape[0], predictions['multixcan_mashr'])
        random_recall, random_precision, random_averages = get_random_classifier_pr(data, reps=100)
#         display(len(random_recall))
        random_label = f'Random - AP: {random_averages.mean():.3f}'
        plot_pr_raw_data(random_recall, random_precision, random_label, estimator='mean', ax=ax, ci='sd', color='gray')

#         ax.set_title('Precision-Recall curves using PharmacotherapyDB')
        #ax.plot([0.00, 1.00], [1.00, 0.00], color='gray', linewidth=0.50)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.60, 1.02])
#         ax.legend(loc="upper right")
        ax.get_legend().remove()
    #     ax.set_aspect('equal')


# %%
plot_pr_for_methods(methods_names)
# plt.savefig('/tmp/pr.pdf', bbox_inches='tight')

# %%
