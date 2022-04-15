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

#########################################################################################################
# 分类器

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "LogisticRegression"})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 100, "max_depth": 5,
         "objective": "multi:softprob", "verbosity": 0, "nthread": -1, "learning_rate": 0.1, "num_class": 2})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


# 构建XGBboost基准分类器
def base_xgb_learner(base_learner_number, n_estimator, max_depth, learning_rate):
    learner = []
    for i in range(base_learner_number):
        print('==========XGB' + str(i + 1) + '==========')
        learner.append(XGBClassifier(n_estimators=500, max_depth=max_depth, learning_rate=learning_rate))
    return learner


#
def base_MLP_learner(base_learner_number):
    learner = []
    for i in range(base_learner_number):
        print('==========MLP' + str(i + 1) + '==========')
        learner.append(MLPClassifier(hidden_layer_sizes=300))
    return learner

def base_gbdt_learner(base_learner_number):
    learner = []
    for i in range(base_learner_number):
        print('==========GBDT' + str(i + 1) + '==========')
        learner.append(GBDT(n_estimators=500, random_state=10))
    return learner

def base_et_learner(base_learner_number):
    learner = []
    # config = get_toy_config()
    for i in range(base_learner_number):
        # if i % 2 == 0 or i >= 5:
        print('==========ETree' + str(i + 1) + '==========')
        learner.append(ET(n_estimators=50, random_state=10))

        # else:
        #     print('==========MLP' + str(i + 1) + '==========')
        #     learner.append(MLPClassifier(hidden_layer_sizes=200))
        # learner.append(GCForest(config))
    return learner


def base_gcforest_learner(base_learner_number):
    learner = []

    for i in range(base_learner_number):
        print('==========gcforest' + str(i + 1) + '==========')

        config = get_toy_config()
        rf = GCForest(config)
        learner.append(rf)

    return learner

