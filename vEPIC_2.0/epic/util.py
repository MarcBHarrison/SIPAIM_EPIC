from math import sqrt

import numpy as np
import scipy as sp
from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from statsmodels.regression.linear_model import OLS

from epic.consts import *


def partition_matrix(partition):
    """
    Create a partition matrix for the dataset using the provided partition row
    """

    labels, labels_index = np.unique(partition, return_inverse=True)
    p = np.zeros((len(partition), len(labels)))
    p[np.arange(0, len(partition)), labels_index] = 1

    for i in range(0, len(labels)):
        p[:, i] = p[:, i] / np.sum(p[:, i])

    return p


def merge_data(partition, dataset):
    """
    Create a merged dataset from dataset using merge_rule
    """
    p = partition_matrix(partition)
    mergeset = np.zeros((dataset.shape[0], p.shape[1]))
    for i in range(0, dataset.shape[0]):
        mergeset[i, :] = dataset[i, :].dot(p)

    return mergeset


def m_sort(x):
    if len(x):
        return x[0]
    return False

def median_diff(dataset, feat_col, dx_col, sex_col):
    m = 0
    f = 1
    cn = 0
    dx = 1

    m_feats = dataset[dataset[sex_col] == m]
    m_cn_feats = m_feats[m_feats[dx_col] == cn]
    m_cn_mean = np.mean(m_cn_feats[feat_col])
    m_cn_sigma = np.std(m_cn_feats[feat_col])
    m_cn_n = m_cn_feats.shape[0]
    m_dx_feats = m_feats[m_feats[dx_col] == dx]
    m_dx_mean = np.mean(m_dx_feats[feat_col])
    m_dx_sigma = np.std(m_dx_feats[feat_col])
    m_dx_n = m_dx_feats.shape[0]

    f_feats = dataset[dataset[sex_col] == f]
    f_cn_feats = f_feats[f_feats[dx_col] == cn]
    f_cn_mean = np.mean(f_cn_feats[feat_col])
    f_cn_sigma = np.std(f_cn_feats[feat_col])
    f_cn_n = f_cn_feats.shape[0]
    f_dx_feats = f_feats[f_feats[dx_col] == dx]
    f_dx_mean = np.mean(f_dx_feats[feat_col])
    f_dx_sigma = np.std(f_dx_feats[feat_col])
    f_dx_n = f_dx_feats.shape[0]

    sex_tdiff = (m_cn_mean - m_dx_mean)/sqrt(m_cn_sigma**2/m_cn_n + m_dx_sigma**2/m_dx_n) - (f_cn_mean - f_dx_mean)/sqrt(f_cn_sigma**2/f_cn_n + f_dx_sigma**2/f_dx_n)

    cn_feats = dataset[dataset[dx_col] == cn]
    cn_sigma = np.std(cn_feats[feat_col])
    cn_mean = np.mean(cn_feats[feat_col])
    cn_n = cn_feats.shape[0]

    dx_feats = dataset[dataset[dx_col] == dx]
    dx_sigma = np.std(dx_feats[feat_col])
    dx_mean = np.mean(dx_feats[feat_col])
    dx_n = dx_feats.shape[0]

    dx_tdiff = (cn_mean - dx_mean)/sqrt(cn_sigma**2/cn_n + dx_sigma**2/dx_n)

    return sex_tdiff, dx_tdiff


def aic(group, train):
    """Akaike Information Criterion"""
    ols = OLS(group, train).fit()
    k = train.shape[1]
    return 2*k - 2*ols.llf


def get_classifier(classifier_name):

    if classifier_name == 'Linear SVM' or classifier_name == 'Linear_SVM':
        clf = svm.LinearSVC(class_weight='balanced', C=1, dual=False,
                            penalty='l1', random_state=1, loss='squared_hinge')
    elif classifier_name == 'RBF SVM' or classifier_name == 'RBF_SVM':
        clf = svm.SVC(kernel='rbf', class_weight='balanced')
    elif classifier_name == 'LDA' or classifier_name == 'lda':
        clf = LinearDiscriminantAnalysis()
    elif classifier_name == 'LR':
        clf = LogisticRegression()

    return clf

def performance_measures(y_true, y_pred):

    tp = np.count_nonzero((y_pred == 1) & (y_true == 1))
    tn = np.count_nonzero((y_pred == 0) & (y_true == 0))
    fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
    fn = np.count_nonzero((y_pred == 0) & (y_true == 1))

    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0

    accuracy = (tp + tn) / (tp + fn + tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2

    f1 = (2 * tp) / (2 * tp + fp + fn)
    if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) == 0:
        mcc = 0
    else:
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    ppv = bool(tp + fp) and (tp)/(tp + fp) or 0

    performance = np.zeros((1, NUM_PERFORMANCE_METRICS))
    performance[0, 0] = balanced_accuracy
    performance[0, 1] = accuracy
    performance[0, 2] = sensitivity
    performance[0, 3] = specificity
    performance[0, 4] = f1
    performance[0, 5] = mcc
    performance[0, 6] = ppv

    return performance
