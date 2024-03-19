import sys
import torch
import  preprocess
import NE
import numpy as np
import pandas as pd
import time
import os
import tools
import autoencoder
import scipy.sparse as sp
import selfsupervised
from sklearn.preprocessing import LabelEncoder
import self
from scipy.stats import pearsonr
import gc
from fbpca import pca
from sklearn.preprocessing import LabelEncoder,scale,MinMaxScaler
import matplotlib.pyplot as plt
if __name__ == '__main__':
    para_set = {}


    def run(para_set):
        if para_set['dataname'] == 'Filtered_Baron_HumanPancreas':
            para_set = {'dataname': 'Filtered_Baron_HumanPancreas', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 5000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 600, 'lr3': 0.002, 'top_k': 50, 'top_k2': 20, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.5,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_MousePancreas':
            para_set = {'dataname': 'Filtered_MousePancreas', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 5000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 500, 'lr3': 0.002, 'top_k': 50, 'top_k2': 30, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.5,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_Muraro_HumanPancreas':
            para_set = {'dataname': 'Filtered_Muraro_HumanPancreas', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 10000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 600, 'lr3': 0.001, 'top_k': 50, 'top_k2': 20, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.5,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_Segerstolpe_HumanPancreas':
            para_set = {'dataname': 'Filtered_Segerstolpe_HumanPancreas', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 5000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 600, 'lr3': 0.002, 'top_k': 50, 'top_k2': 30, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.5,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_DownSampled_SortedPBMC':
            para_set = {'dataname': 'Filtered_DownSampled_SortedPBMC', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 5000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 600, 'lr3': 0.004, 'top_k': 50, 'top_k2': 30, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.4,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_68K_PBMC':
            para_set = {'dataname': 'Filtered_68K_PBMC', 'proprecess': 'int', 'method': 'corrcoef', 'epoch1': 5000,
                        'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 500, 'lr3': 0.002, 'top_k': 40, 'top_k2': 30, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.4,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 4}
        elif para_set['dataname'] == 'Filtered_Xin_HumanPancreas':
            para_set = {'dataname': 'Filtered_Xin_HumanPancreas', 'proprecess': 'int', 'method': 'corrcoef',
                        'epoch1': 5000, 'lr1': 0.00001, 'epoch2': 20, 'lr2': 0.00005,
                        'epoch3': 500, 'lr3': 0.002, 'top_k': 50, 'top_k2': 20, 'hidden': 256, 'hidden2': 32,
                        'dropout': 0.4,
                        'num_layers': 3, 'lam': 1, 'concat': 1, 'samples': 1}
        data_path = 'D:/yuxianhai/data/'#'F:/@代码/代码备份/data/'  # 'D:/YXH/data/'#'D:/yuxianhai/data/'

        data = pd.read_csv(data_path + para_set['dataname'] + '_data.csv', index_col=0)
        data = data.values
        labels = pd.read_csv(data_path + para_set['dataname'] + '_Labels.csv')
        labels = labels.values
        print(data.shape)
        print(labels.shape)

        #data, labels = tools.clean_10(data, labels)

        label = np.unique(labels)  # 去除重复元素，以计算类别数目
        num_label = len(label)

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #data = scaler.fit_transform(data)
        #train_data_all = data / np.max(data)
        data, _ = tools.high_var_npdata(data, num=1000, ind=1)

        is_large = data.shape[0] > 10000
        if not is_large:
            if os.path.exists(para_set['dataname'] + 'similarity_matrix.csv'):
                matrix = pd.read_csv(para_set['dataname'] + 'similarity_matrix.csv', header=None, sep=',')
            else:
                matrix = np.corrcoef(data, dtype=np.float32)
                matrix = matrix.astype(np.float16)
                matrix = pd.DataFrame(matrix)
                #np.savetxt(para_set['dataname'] + 'similarity_matrix.csv', matrix, fmt='%.6f', delimiter=",")
            # if para_set['method'] == 'Gaussian':
            #     matrix = tools.getmatrix(data, para_set)
            #     os.remove('test.ann')


            adj = tools.getgraph1(matrix, para_set['top_k2'])  # csr_matrix类型
            features = np.asarray(data)  # , 'float32'
            edge_index = tools.getgraph2(matrix,para_set['top_k'])
            del matrix
            del data
            gc.collect()
        else:
            features = data.astype('float32')
            adj = tools.getgraph3(features, para_set['top_k2'])
            edge_index = tools.getgraph3(features,para_set['top_k'])

            # adj = tools.getgraph1(matrix, para_set)  # csr_matrix类型
            features = np.asarray(features)  # , 'float32'
            #features, val, vec = pca(data, para_set['pca_dim'])  # PCA返回降维结果、特征值、特征向量
            features = features.astype('float16')
            del data
            gc.collect()

        le = LabelEncoder()
        le.fit(labels)
        real_label = le.transform(labels)
        features = np.log1p(features)
        maxscale = np.max(features)
        print('maxscale:', maxscale)
        features = features / np.max(features)
        for k in range(10):
            acc_list = []
            f1_list = []
            for i in range(5):
                print('this is No.{:d} times------------\n'.format(i + 1))
                #if not os.path.exists(para_set['dataname'] + 'cvae.pkl'):
                autoencoder.GraphAutoEncoder(adj, features, para_set, is_large)
                # selfsupervised.pretrain(edge_index, features, para_set, i, first=True)
                # selfsupervised.pretrain(edge_index, features, para_set, i, first=False)
                # acc_1, f1_1,Train_Loss_list_1,val_acc_list_1 = selfsupervised.finetune_with_pretrain(edge_index, features, real_label, num_label, para_set, i + 1)
                # torch.cuda.empty_cache()
                acc, f1,Train_Loss_list_2,val_acc_list_2 = selfsupervised.finetune_without_pretrain(edge_index, features, real_label, num_label, para_set, i + 1,1000)
                acc_list.append(acc)
                f1_list.append(f1)
            avg_acc = round(sum(acc_list) / len(acc_list), 6)
            avg_f1 = round(sum(f1_list) / len(f1_list), 6)
            with open('record.txt', 'a') as f:
                string = str(para_set['dataname']) + '\t' + \
                         str(para_set['proprecess']) + '\t' + \
                         str(para_set['method']) + '\t' + \
                         str('without pretrain') + '\t' + \
                         str(acc_list[0]) + '\t' + str(acc_list[1]) + '\t' + str(acc_list[2]) + '\t' + str(
                    acc_list[3]) + '\t' + str(acc_list[4]) + '\t' + \
                         str(f1_list[0]) + '\t' + str(f1_list[1]) + '\t' + str(f1_list[2]) + '\t' + str(
                    f1_list[3]) + '\t' + str(f1_list[4]) + '\t' + \
                         str("avg_acc") + '\t' + str(avg_acc) + '\t' + str("avg_f1") + '\t' + str(avg_f1) + '\t' + \
                         str(para_set['epoch1']) + '\t' + \
                         str(para_set['lr1']) + '\t' + \
                         str(para_set['epoch2']) + '\t' + \
                         str(para_set['lr2']) + '\t' + \
                         str(para_set['epoch3']) + '\t' + \
                         str(para_set['lr3']) + '\t' + \
                         str(para_set['top_k']) + '\t' + \
                         str(para_set['top_k2']) + '\t' + \
                         str(para_set['hidden']) + '\t' + \
                         str(para_set['hidden2']) + '\t' + \
                         str(para_set['dropout']) + '\t' + \
                         str(para_set['num_layers']) + '\t' + \
                         str(para_set['lam']) + '\t' + \
                         str(para_set['concat']) + '\t' + \
                         str(para_set['samples'])
                f.write(string)
                f.write('\r\n')


    dataname = ['Filtered_Baron_HumanPancreas']  # 'Filtered_DownSampled_SortedPBMC','Filtered_Segerstolpe_HumanPancreas','Filtered_Baron_HumanPancreas','Filtered_Muraro_HumanPancreas','Filtered_MousePancreas',

    for i in range(len(dataname)):
        para_set['dataname'] = dataname[i]
        run(para_set)