import numpy as np
def base_learners_results(metric_dict, fold_num, group_num, f):
    for i in range(group_num):
        ave_acc = 0
        ave_prec = 0
        ave_recall = 0
        ave_f1_score = 0
        ave_auc = 0
        ave_sum = 0
        bl_metric_list = []
        for fold in range(fold_num):
            temp_list = metric_dict[fold]
            bl_metric_list.append(temp_list[i])
        bl_metric_list = np.array(bl_metric_list)
        ave_acc = np.mean(bl_metric_list[:,0])
        ave_prec = np.mean(bl_metric_list[:,1])
        ave_recall = np.mean(bl_metric_list[:,2])
        ave_f1_score = np.mean(bl_metric_list[:,3])
        ave_auc = np.mean(bl_metric_list[:,4])
        ave_aupr = np.mean(bl_metric_list[:,5])
        f.write('the '+ str(i+1)+ ' base learner proformance: \tAcc\t'+ str(ave_acc)+'\tprec\t'+ str(ave_prec)+ '\trecall\t'+str(ave_recall)+'\tf1_score\t'+str(ave_f1_score)+'\tAUC\t'+ str(ave_auc)+'\tAUPR\t'+ str(ave_aupr)+'\n')
