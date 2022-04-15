import numpy as np
import random
# 构建正标签样本和无标签样本
def circRNA_disease_vector(dataset):
    print("==========构建正标签样本和无标签样本==========")
    print("==========import circRNA similarity==========")
    circRNA_simlarity = np.loadtxt(dataset + "/c-c-f.csv", delimiter=",")
    print("==========import disease similarity==========")
    disease_simlarity = np.loadtxt(dataset + "/d-d.csv", delimiter=",")
    print("==========import circRNA-disease association matrix==========")
    ass_matrix = np.loadtxt(dataset + "/ass matrix.csv", delimiter=",")
    print("==========import miRNA similarity==========")
    miRNA_similarity = np.loadtxt(dataset + "/m-m.csv", delimiter=",")
    miRNA_similarity_ave = miRNA_similarity[miRNA_similarity.shape[0] - 1, :]
    miRNA_similarity = miRNA_similarity[:-1, :]
    print("==========import circRNA-miRNA association matrix==========")
    c_m = np.loadtxt(dataset + "/c-m.csv", delimiter=",")
    print("==========import miRNA-disease association matrix==========")
    m_d = np.loadtxt(dataset + "/m-d.csv", delimiter=",")
    print("==========construct positive pairs and unlabelled pairs==========")

    # 环状RNA和疾病之间
    # positive_list = []
    # negative_list = []
    # for i in range(circRNA_simlarity.shape[0]):
    #     for j in range(miRNA_similarity.shape[0]):
    #         if c_m[i][j] == 1:
    #             positive_cm = circRNA_simlarity[i,:].tolist()+ miRNA_similarity[j,:].tolist()
    #             for k in range(disease_simlarity.shape[0]):
    #                 if ass_matrix[i,k] == 1:
    #                     positive_cmd = positive_cm + disease_simlarity[k,:].tolist()+ [1, 0]
    #                     positive_list.append(positive_cmd)
    #                 else:
    #                     negative_cmd = positive_cm + disease_simlarity[k,:].tolist()+ [0,1]
    #                     negative_list.append(negative_cmd)
    #         else:
    #             negative_cm = circRNA_simlarity[i,:].tolist()+ miRNA_similarity_ave.tolist()
    #             for t in range(disease_simlarity.shape[0]):
    #                 if ass_matrix[i,t] == 1:
    #                     n_positive_cmd = negative_cm + disease_simlarity[t,:].tolist()+ [1, 0]
    #                     positive_list.append(n_positive_cmd)
    #                 else:
    #                     n_negative_cmd = negative_cm + disease_simlarity[t,:].tolist()+ [0,1]
    #                     negative_list.append(n_negative_cmd)

    known = []
    known_index = 0
    unknown = []
    unknown_index = 0
    for i in range(circRNA_simlarity.shape[0]):
        for j in range(disease_simlarity.shape[0]):
            if ass_matrix[i, j] == 1:
                known.append((i, j))
                known_index = known_index + 1
            else:
                unknown.append((i, j))
                unknown_index = unknown_index + 1

    # 环状RNA和miRNA之间
    # known_cm = []
    # known_index_cm = 0
    # unknown_cm = []
    # unknown_index_cm = 0

    # for i in range(circRNA_simlarity.shape[0]):
    #     for j in range(miRNA_similarity.shape[0]):
    #         if c_m[i, j] == 1:
    #             known_cm.append((i, j))
    #             known_index_cm = known_index_cm + 1
    #         else:
    #             unknown_cm.append((i, j))
    #             unknown_index_cm = unknown_index_cm + 1

    circRNA_miRNA_similarity = np.zeros(
        (circRNA_simlarity.shape[0], circRNA_simlarity.shape[0] + miRNA_similarity.shape[0]))
    circRNA_index = []
    for i in range(circRNA_simlarity.shape[0]):
        for j in range(miRNA_similarity.shape[0]):
            if c_m[i, j] == 1:
                if i not in circRNA_index:
                    circRNA_miRNA_similarity[i, :] = np.concatenate((circRNA_simlarity[i, :], miRNA_similarity[j, :]))
                    circRNA_index.append(i)
                else:
                    index = 0
                    for x in circRNA_index:
                        if x == i:
                            index = index + 1
                    mirna_simlarity_pre = circRNA_miRNA_similarity[i, circRNA_miRNA_similarity.shape[0]:]
                    mirna_simlarity_now = (mirna_simlarity_pre * index + miRNA_similarity[j, :]) / (index + 1)
                    circRNA_miRNA_similarity[i, :] = np.concatenate((circRNA_simlarity[i, :], mirna_simlarity_now))
            else:
                circRNA_miRNA_similarity[i, :] = np.concatenate((circRNA_simlarity[i, :], miRNA_similarity_ave))

    # cm_list = []
    # for i in range(known_index_cm):
    #     positive = circRNA_simlarity[known_cm[i][0], :].tolist() + miRNA_similarity[known_cm[i][1], :].tolist() + [known_cm[i][0]]
    #     cm_list.append(positive)
    # for j in range(unknown_index_cm):
    #     negative = circRNA_simlarity[unknown_cm[i][0],:].tolist()+miRNA_similarity_ave.tolist()+[unknown_cm[i][0]]
    #     cm_list.append(negative)

    # miRNA和疾病之间
    # known_md = []
    # known_index_md = 0
    # unknown_md = []
    # unknown_index_md = 0
    # for i in range(miRNA_similarity.shape[0]):
    #     for j in range(disease_simlarity.shape[0]):
    #         if m_d[i, j] == 1:
    #             known_md.append((i, j))
    #             known_index_md = known_index_md + 1
    #         else:
    #             unknown_md.append((i, j))
    #             unknown_index_md = unknown_index_md + 1

    positive_list = []
    unlabel_list = []
    for i in range(known_index):
        posi = circRNA_miRNA_similarity[known[i][0], :].tolist() + disease_simlarity[known[i][1], :].tolist() + [1, 0]
        positive_list.append(posi)

    for j in range(unknown_index):
        unlabel = circRNA_miRNA_similarity[unknown[j][0], :].tolist() + disease_simlarity[unknown[j][1], :].tolist() + [
            0, 1]
        unlabel_list.append(unlabel)

    random.shuffle(positive_list)
    random.shuffle(unlabel_list)
    return positive_list, unlabel_list, circRNA_simlarity.shape[0] + disease_simlarity.shape[0] + \
           miRNA_similarity.shape[0],unknown,circRNA_miRNA_similarity,disease_simlarity


# 构建和正样本相同数量的负样本
def constr_same_num_unlabel_data(unlabel_list, cv_num):
    print("==========构建和正样本相同数量的负样本==========")
    unlabel_train = unlabel_list[cv_num:]
    unlabel_cv = unlabel_list[:cv_num]
    return unlabel_train, unlabel_cv

# 构架基准数据特征 resampling
def cons_basetrain_data(postive_train_data, unlabel_train_data):
    samples = postive_train_data.shape[0]
    base_samples = random.sample(unlabel_train_data, samples)
    base = np.concatenate((postive_train_data, base_samples))
    return base