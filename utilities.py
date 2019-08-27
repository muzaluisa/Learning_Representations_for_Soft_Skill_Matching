"""
@author: Luiza Sayfullina
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def find_recall_for_fixed_precision(y_true, pred, desired_precision=0.95):

    """
    :param y_true: true labels
    :param pred: predicted labels
    :param desired_precision: desired precision
    :return: precision, recall, f1_weighted, f1
    """

    precisions = []
    recalls = []
    f1w = []
    f1_ = []
    y_true = np.array(y_true)
    pred = np.array(pred)

    for border in range(-200, 200, 2):
        b = border / 100.0
        f1_weighted = f1_score(y_true, pred > b,average='weighted')
        f1 = f1_score(y_true, pred > b)
        precision = precision_score(y_true,pred>b)
        recall = recall_score(y_true,pred>b)
        precisions.append(precision)
        recalls.append(recall)
        f1w.append(f1_weighted)
        f1_.append(f1)
        if precision - desired_precision > 0:
            break
    if precision == 0:
        ind = [-i for i in range(len(precisions)) if precisions[-i]>0][0]
        print('precision = 0',f1_weighted, precision, recall)
        return round(precisions[ind], 4), round(recalls[ind], 4), round(f1_[ind], 4), round(f1w[ind], 4)

    return round(precision, 4), round(recall, 4), round(f1_weighted, 4), round(f1, 4)


if __name__ == '__main__':

    pass

    # if test_cv:
    #     desired_precision = 0.9
    # else:
    #    desired_precision = 0.95
    #
    # precision, recall, f1_w, f1  = find_recall_for_fixed_precision(y_true_border, pred_border, desired_precision)
    # print('border:', precision, recall, f1_w)