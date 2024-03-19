from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import math
import time

class similarity(object):
    def __init__(self,num_of_neighbor):
        self.num_of_neighbor = int(num_of_neighbor)

    def nearest_neighbor_search(self,data):
        start_time = time.time()
        K = self.num_of_neighbor
        n ,d = data.shape
        t = AnnoyIndex(d ,'euclidean')
        for i in range(n):  # 将所有数据集样本特征顺序添加到索引对象中
            t.add_item(i ,data[i ,:])
        t.build(200)  # build(n_trees) 接口中指定棵数。annoy通过构建一个森林(类似随机森林的思想)来提高查询的精准度，减少方差。
        t.save('test.ann')
        u = AnnoyIndex(d,'euclidean')
        u.load('test.ann')
        index = np.zeros((n, K))
        value = np.zeros((n ,K))
        #distance1 = np.sqrt(np.sum((data[74] - data[252]) ** 2))
        for i in range(n):
            tmp, tmp1 = u.get_nns_by_item(i ,K, include_distances=True)# a.get_nns_by_item(i, n, search_k=-1, include_distances=False)返回第i 个item的n个最近邻的item。在查询期间，它将检索多达search_k（默认n_trees * n）个点。search_k为您提供了更好的准确性和速度之间权衡。如果设置include_distances为True，它将返回一个包含两个列表的2元素元组：第二个包含所有对应的距离。
            index[i ,:] = tmp  # 相似细胞索引
            value[i ,:] = tmp1  # 相似度
            print("\r#1 get nearest neighbors------------{:.2f}%------------".format(100 * i / n), end='')
        print('\r------------get nearest neighbors!  took %f seconds in total------------\n' % (time.time() - start_time))
        #distance_s1 = np.sum(value[74])
        return index.astype('int'), value

    def Gaussian(self,data,index,value,i):
        n, d = data.shape
        Gi = np.zeros(n)
        K = self.num_of_neighbor
        t = AnnoyIndex(d,'euclidean')
        t.load('test.ann')
        u1 = (np.sum(value[i]))/K
        for j in range(n):
            u2 = (np.sum(value[j]))/K
            sigma = 1
            u = sigma*(u1 + u2)/2
            Gi[j] = (math.exp(-1*((t.get_distance(i,j))**2)/(2*u*u)))/(u*math.sqrt(2*math.pi))
            #distance = np.sqrt(np.sum((data[i] - data[j]) ** 2))  #, np.linalg.norm(data[i] - data[j])
            #Gi[j] = math.exp(-1 * (distance ** 2) / (2 * sigma * sigma))
        return Gi

    def sim_matrix(self,data):
        start_time = time.time()
        n, d = data.shape
        index,value = self.nearest_neighbor_search(data)
        G = []
        for i in range(n):
            Gi = self.Gaussian(data,index,value,i)
            print("\r#2 get similarity matrix------------{:.2f}%------------".format(100*i/n), end='')
            G.append(Gi)
        print('\r------------get similarity matrix!  took %f seconds in total------------\n' % (time.time() - start_time))
        return G