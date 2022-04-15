import time
import numpy as np
import random
import argparse
from keras.layers import Dense, Input
from keras.models import Sequential, model_from_config, Model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier as GBDT, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn import neighbors
from sklearn import svm
from sklearn import neighbors
from gcforest.gcforest import GCForest
from sklearn.cluster import KMeans


# performance evaluation

def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] == 1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn) / num
    try:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        recall = float(tp) / (tp + fn)
        f1_score = float((2 * precision * recall) / (precision + recall))
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision = recall = f1_score = 100
    AUC = roc_auc_score(y_test, y_prob)
    p, r, _ = precision_recall_curve(y_test, y_prob)
    AUPR = auc(r, p)
    return tp, fp, tn, fn, acc, precision, recall, f1_score, AUC, AUPR
