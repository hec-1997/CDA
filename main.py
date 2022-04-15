'''
/**
 *          .,:,,,                                        .::,,,::.
 *        .::::,,;;,                                  .,;;:,,....:i:
 *        :i,.::::,;i:.      ....,,:::::::::,....   .;i:,.  ......;i.
 *        :;..:::;::::i;,,:::;:,,,,,,,,,,..,.,,:::iri:. .,:irsr:,.;i.
 *        ;;..,::::;;;;ri,,,.                    ..,,:;s1s1ssrr;,.;r,
 *        :;. ,::;ii;:,     . ...................     .;iirri;;;,,;i,
 *        ,i. .;ri:.   ... ............................  .,,:;:,,,;i:
 *        :s,.;r:... ....................................... .::;::s;
 *        ,1r::. .............,,,.,,:,,........................,;iir;
 *        ,s;...........     ..::.,;:,,.          ...............,;1s
 *       :i,..,.              .,:,,::,.          .......... .......;1,
 *      ir,....:rrssr;:,       ,,.,::.     .r5S9989398G95hr;. ....,.:s,
 *     ;r,..,s9855513XHAG3i   .,,,,,,,.  ,S931,.,,.;s;s&BHHA8s.,..,..:r:
 *    :r;..rGGh,  :SAG;;G@BS:.,,,,,,,,,.r83:      hHH1sXMBHHHM3..,,,,.ir.
 *   ,si,.1GS,   sBMAAX&MBMB5,,,,,,:,,.:&8       3@HXHBMBHBBH#X,.,,,,,,rr
 *   ;1:,,SH:   .A@&&B#&8H#BS,,,,,,,,,.,5XS,     3@MHABM&59M#As..,,,,:,is,
 *  .rr,,,;9&1   hBHHBB&8AMGr,,,,,,,,,,,:h&&9s;   r9&BMHBHMB9:  . .,,,,;ri.
 *  :1:....:5&XSi;r8BMBHHA9r:,......,,,,:ii19GG88899XHHH&GSr.      ...,:rs.
 *  ;s.     .:sS8G8GG889hi.        ....,,:;:,.:irssrriii:,.        ...,,i1,
 *  ;1,         ..,....,,isssi;,        .,,.                      ....,.i1,
 *  ;h:               i9HHBMBBHAX9:         .                     ...,,,rs,
 *  ,1i..            :A#MBBBBMHB##s                             ....,,,;si.
 *  .r1,..        ,..;3BMBBBHBB#Bh.     ..                    ....,,,,,i1;
 *   :h;..       .,..;,1XBMMMMBXs,.,, .. :: ,.               ....,,,,,,ss.
 *    ih: ..    .;;;, ;;:s58A3i,..    ,. ,.:,,.             ...,,,,,:,s1,
 *    .s1,....   .,;sh,  ,iSAXs;.    ,.  ,,.i85            ...,,,,,,:i1;
 *     .rh: ...     rXG9XBBM#M#MHAX3hss13&&HHXr         .....,,,,,,,ih;
 *      .s5: .....    i598X&&A&AAAAAA&XG851r:       ........,,,,:,,sh;
 *      . ihr, ...  .         ..                    ........,,,,,;11:.
 *         ,s1i. ...  ..,,,..,,,.,,.,,.,..       ........,,.,,.;s5i.
 *          .:s1r,......................       ..............;shs,
 *          . .:shr:.  ....                 ..............,ishs.
 *              .,issr;,... ...........................,is1s;.
 *                 .,is1si;:,....................,:;ir1sr;,
 *                    ..:isssssrrii;::::::;;iirsssssr;:..
 *                         .,::iiirsssssssssrri;;:.
 */
'''


"""
@author: Hec
@Title:a ensemble learning method based on stacked auto encoder and gcforest with muti_data fusion to predict 
       potential circRNA-disease associations.
"""

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
from classfiers import *
from kmeans_negative import *
from evaluate import *
from soft_voting import *
from stacked_autoencoder import *
from ELCDA.utils import base_learners_results



parse = argparse.ArgumentParser(description="ensemble learning for circRNA disease associations prediction")
parse.add_argument("-dataset", default="CircR2Disease", type=str,
                   help="choose which dataset:CircR2Disease,Three Datasets .el")
parse.add_argument("-base_learner_num", default=12, type=int,
                   help="base learner number,eg:10,11,12,13..")
parse.add_argument("-k_fold", default=5, type=int,
                   help="K_fold crss validation,eg:2,5,10..")

args = parse.parse_args()

now = time.asctime(time.localtime(time.time()))
positive_list, unlabel_list, rna_dis_num,unknown,circRNA_miRNA_similarity,disease_simlarity = circRNA_disease_vector(args.dataset)
unlabel_train, unlabel_cv = constr_same_num_unlabel_data(unlabel_list, len(positive_list))
base_learner_num = args.base_learner_num


metric_dict = {}
soft_acc_list = []
soft_prec_list = []
soft_recall_list = []
soft_f1_score_list = []
soft_auc_list = []
soft_aupr_list = []

train_sets = []
test_sets = []
with open('./results/ELCDA process results_3.txt', 'a') as f:
    f.write('\n\n**************************************************************\n' + now + '\n')
    f.write('cross validation fold number\t' + str(args.k_fold) + '\tbase_learner_num\t' + str(
        base_learner_num)  + '\n')
    for fold in range(args.k_fold):
        f.write('\n-------------------------Fold' + str(fold + 1) + '---------------------------\n')
        print("==========Fold" + str(fold + 1) + "==========")
        positive_train_data = np.array([x for i, x in enumerate(positive_list) if i % args.k_fold != 0])
        positive_test_data = np.array([x for i, x in enumerate(positive_list) if i % args.k_fold == 0])
        negative_test_data = np.asarray([x for i, x in enumerate(unlabel_cv) if i % args.k_fold == 0])
        X_test = np.concatenate((positive_test_data[:, :-2], negative_test_data[:, :-2]))
        Y_test = np.concatenate((positive_test_data[:, -2], negative_test_data[:, -2]))
        # clfs = base_et_learner(args.base_learner_num)  # 1
        clfs = base_gcforest_learner(args.base_learner_num)  # 1

        metric_list = []
        trained_clfs = []
        most_imps_list = []
        encoder1_list = []
        for i in range(args.base_learner_num):
            print("==========  " + str(i + 1) + "  base learner running==========")
            # base_train = cons_basetrain_data(positive_train_data, unlabel_train)
            base_train = k_means_negative(positive_train_data,unknown,circRNA_miRNA_similarity,disease_simlarity)
            x_base_train = np.array(base_train[:, :-2])
            y_base_train = np.array(base_train[:, -2])
            x_base_test = X_test

            prefilter_train_bef, prefilter_test_bef, encoder1 = autoencoder_fine_tuning(x_base_train, y_base_train,
                                                                                        x_base_test, Y_test)
            encoder1_list.append(encoder1)

            print("trainning shape ==> x_base_train shape:", prefilter_train_bef.shape)
            print("trainning shape ==> y_base_train shape:", prefilter_test_bef.shape)

            clf = clfs[i]
            # clf.fit(prefilter_train_bef, y_base_train)
            y_base_train = y_base_train.flatten()
            clf.fit_transform(prefilter_train_bef, y_base_train)
            trained_clfs.append(clf)

            print("testing shape ==> x_base_test shape:", prefilter_test_bef.shape)
            print("testing shape ==> y_base_test shape:", Y_test.shape)
            # 预测为某个标签
            y_pred = clf.predict(prefilter_test_bef)
            # 预测为某个标签的额概率
            y_pred_prob = clf.predict_proba(prefilter_test_bef)
            # 去预测为1的概率
            y_pred_prob = y_pred_prob[:, 1]
            tp, fp, tn, fn, acc, prec, recall, f1_score, AUC, aupr = calculate_performace(len(y_pred), y_pred,
                                                                                          y_pred_prob, Y_test)
            print('the ', i + 1, ' base learner proformance: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t',
                  recall, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  aupr = \t', aupr)

            f.write('the ' + str(i + 1) + ' base learner proformance: \t  tp = \t' + str(tp) + '\t fp = \t' + str(
                fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(fn) + '\t  Acc = \t' + str(
                acc) + '\t  prec = \t' + str(prec) + '\t  recall = \t' + str(recall) + '\t  f1_score = \t' + str(
                f1_score) + '\t  AUC = \t' + str(auc) + '\t  AUPR = \t' + str(aupr) + '\n')

            metric_list.append([acc, prec, recall, f1_score, AUC, aupr])

        metric_dict[fold] = metric_list

        base_preds, base_probs = base_preds_probs_imp_by_encoder(X_test, encoder1_list, trained_clfs)
        # base_preds, base_probs = base_preds_probs(x_base_test, trained_clfs)
        pred_final, prob_final = soft_voting_strategy(base_probs)
        soft_tp, soft_fp, soft_tn, soft_fn, soft_acc, soft_prec, soft_recall, soft_f1_score, soft_auc, soft_aupr = calculate_performace(
            len(pred_final), pred_final, prob_final, Y_test)
        print('Auto_Encoder_Gcforest soft voting proformance: \n  Acc = \t', soft_acc, '\n  prec = \t', soft_prec, '\n  recall = \t',
              soft_recall, '\n  f1_score = \t', soft_f1_score, '\n  AUC = \t', soft_auc,
              '\n  AUPR = \t', soft_aupr)
        f.write('Auto_Encoder_Gcforest soft voting proformance: \ttp\t' + str(soft_tp) + '\tfp\t' + str(soft_fp) + '\ttn\t' + str(
            soft_tn) + '\tfn\t' + str(soft_fn) + '\tAcc\t' + str(soft_acc) + '\tprec\t' + str(
            soft_prec) + '\trecall\t' + str(soft_recall) + '\tf1_score\t' + str(soft_f1_score) + '\tAUC\t' + str(
            soft_auc) + '\tAUPR\t' + str(soft_aupr) + '\n')

        soft_acc_list.append(soft_acc)
        soft_prec_list.append(soft_prec)
        soft_recall_list.append(soft_recall)
        soft_f1_score_list.append(soft_f1_score)

        soft_auc_list.append(soft_auc)
        soft_aupr_list.append(soft_aupr)

    print('============================================================')
    soft_acc_arr = np.array(soft_acc_list)
    soft_prec_arr = np.array(soft_prec_list)
    soft_recall_arr = np.array(soft_recall_list)
    soft_f1_score_arr = np.array(soft_f1_score_list)

    soft_auc_arr = np.array(soft_auc_list)
    soft_aupr_arr = np.array(soft_aupr_list)

    soft_ave_acc = np.mean(soft_acc_arr)
    soft_ave_prec = np.mean(soft_prec_arr)
    soft_ave_recall = np.mean(soft_recall_arr)
    soft_ave_f1_score = np.mean(soft_f1_score_arr)

    soft_ave_auc = np.mean(soft_auc_arr)
    soft_ave_aupr = np.mean(soft_aupr_arr)

    soft_std_acc = np.std(soft_acc_arr)
    soft_std_prec = np.std(soft_prec_arr)
    soft_std_recall = np.std(soft_recall_arr)
    soft_std_f1_score = np.std(soft_f1_score_arr)

    soft_std_auc = np.std(soft_auc_arr)
    soft_std_aupr = np.std(soft_aupr_arr)
    f.write('\n------------------- the final results of CDA - ' + now + ' -------------------\n')
    base_learners_results(metric_dict, args.k_fold, base_learner_num, f)
    print('Auto_Encoder_Gcforest Final proformance: \n  Acc = ', soft_ave_acc, '\n  prec = ', soft_ave_prec, '\n  recall = ',
          soft_ave_recall, '\n  f1_score = ', soft_ave_f1_score, '\n  AUC = ', soft_ave_auc,
          '\n AUPR = ', soft_ave_aupr)
    f.write('Auto_Encoder_Gcforest Final proformance: ' +  '\tbase_learner_num\t' + str(base_learner_num) + '\tAcc\t' + str(soft_ave_acc) + '&' + str(
        soft_std_acc) + '\tprec\t' + str(soft_ave_prec) + '&' + str(soft_std_prec) + '\trecall\t' + str(
        soft_ave_recall) + '&' + str(soft_std_recall) + '\tf1_score\t' + str(soft_ave_f1_score) + '&' + str(
        soft_std_f1_score) + '\tAUC\t' + str(soft_ave_auc) + '&' + str(soft_std_auc) + '\tAUPR\t' + str(
        soft_ave_aupr) + '&' + str(soft_std_aupr))
    f.close()
