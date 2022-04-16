import pandas as pd
import numpy as np
import copy
from sklearn.metrics import f1_score,roc_auc_score

def macro_f1_val_helper(labels: np.array, preds: np.array, weights, verbose=True,
                        ):
    df = pd.DataFrame({
        'labels': labels,
        'preds': preds
    })

    # weights = [3 / 7, 2 / 7, 1 / 7, 1 / 7]

    macro_F1 = 0.
    for i in range(len(weights)):
        TP = len(df[(df['labels'] == i) & (df['preds'] == i)])
        FP = len(df[(df['labels'] != i) & (df['preds'] == i)])
        FN = len(df[(df['labels'] == i) & (df['preds'] != i)])
        precision = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        F1 = 2 * precision * recall / (precision + recall + 1e-20)
        macro_F1 += weights[i] * F1
        if verbose is True:
            print(f"class:{i} precision:{precision:.4f} recall:{recall:.4f} F1:{F1:.4f}")
    # Fisrt Stage
    return macro_F1


def macro_f1_val(labels: np.array, preds: np.array, verbost=True):
    overall_F1 = macro_f1_val_helper(labels, preds, weights=[3 / 7, 2 / 7, 1 / 7, 1 / 7])

    labels_first = copy.deepcopy(labels)
    preds_first = copy.deepcopy(preds)
    labels_first[labels_first == 1] = 0
    labels_first[labels_first == 2] = 1
    labels_first[labels_first == 3] = 2

    preds_first[preds_first == 1] = 0
    preds_first[preds_first == 2] = 1
    preds_first[preds_first == 3] = 2

    first_F1 = macro_f1_val_helper(labels_first, preds_first, weights=[5 / 7, 1 / 7, 1 / 7])

    # labels_second=copy.deepcopy(labels)
    # preds_second=copy.deepcopy(preds)
    labels_second = labels[(labels == 0) | (preds == 0) | (labels == 1) | (preds == 1)]
    preds_second = preds[(labels == 0) | (preds == 0) | (labels == 1) | (preds == 1)]

    second_F1=macro_f1_val_helper(labels_second,preds_second,weights=[3/5,2/5])

    return overall_F1,first_F1,second_F1