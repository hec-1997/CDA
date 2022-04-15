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
from data_process import *

# voting strategy & auto_encoder, return preds and probs of base learners
def base_preds_probs_imp_by_encoder(X_test, encoder1_list, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    count = 0
    for clf in trained_clfs:
        X_test_si = np.copy(X_test)
        # most_imp = most_imps_list[count]
        for ae in encoder1_list[count]:
            X_test_si = ae.predict(X_test_si)
        # X_test_si = X_test[:, most_imp]
        pred = clf.predict(X_test_si)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test_si)
        prob_list.append(prob[:, 1])
        count = count + 1
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:, i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:, i])

    return base_preds, base_probs


# voting strategy & feature selection, return preds and probs of base learners
def base_preds_probs_imp_by_rf(X_test, most_imps_list, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    count = 0
    for clf in trained_clfs:
        most_imp = most_imps_list[count]
        X_test_si = X_test[:, most_imp]
        pred = clf.predict(X_test_si)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test_si)
        prob_list.append(prob[:, 1])
        count = count + 1
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:, i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:, i])

    return base_preds, base_probs


# 软投票集成
def soft_voting_strategy(base_probs):
    print('==========Soft voting==========')
    pred_final = []
    prob_final = []
    for prob in base_probs:
        mean_prob = np.mean(prob)
        prob_final.append(mean_prob)
        if mean_prob > 0.5:
            pred_final.append(1)
        else:
            pred_final.append(0)
    return pred_final, prob_final


# voting strategy, return preds and probs of base learners
def base_preds_probs(X_test, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    count = 0
    for clf in trained_clfs:
        pred = clf.predict(X_test)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test)
        prob_list.append(prob[:, 1])
        count = count + 1
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:, i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:, i])

    return base_preds, base_probs