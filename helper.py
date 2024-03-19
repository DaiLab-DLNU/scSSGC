#import matplotlib
#from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# fig = plt.figure()
# #创建3d绘图区域
# ax = plt.axes(projection='3d')
# data = pd.read_csv('data/similarity_matrix.csv',header=None, sep=',')
# data = data.values
# print(type(data))
# print((data.shape))
# x = np.arange(0, 1999, 1)
# y = np.arange(0, 1999, 1)
# X, Y = np.meshgrid(x, y)
# # print(X)
# # print(Y)
# z = data[x][y]
#
# ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
# plt.draw()
# plt.show()
# plt.savefig('3D.jpg')

# data = pd.read_csv('data/data.txt', sep='\t', index_col=0).T
# data = data.values
# sw = []
# sw.append(data[75])
# sw.append(data[1790])
# sw.append(data[1782])
# sw.append(data[4873])
# sw.append(data[4836])
# sw.append(data[4009])
# aa = sw
# np.savetxt('data/sw_matrix2.csv', sw, fmt='%f', delimiter=",")
# import time
# n=3000
# for i in range(n):
#     print("\r{:3f}%".format(i / n), end='')
#     time.sleep(0.01)

# import csv
#
# # Open tsv and txt files(open txt file in write mode)
# tsv_file = open("data/data.tsv")
# txt_file = open("data/data.txt", "w")
#
# # Read tsv file and use delimiter as \t. csv.reader
# # function retruns a iterator
# # which is stored in read_csv
# read_tsv = csv.reader(tsv_file, delimiter="\t")
#
# # write data in txt file line by line
# for row in read_tsv:
#     joined_string = "\t".join(row)
#     txt_file.writelines(joined_string + '\n')
#
# # close files
# txt_file.close()
import torch
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import train_test_split_edges
# import numpy as np
#
# dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
# print(dataset.data)
# data = dataset[0]
# data = train_test_split_edges(data)
# print(data)
# a = data.train_pos_edge_index
# print(a)
# np.savetxt('data/111111.csv',a, fmt='%.2f', delimiter=",")
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# data_path = 'data/'
# lables = pd.read_csv(data_path + 'cleaned_lables.csv', header=None, sep=',')
# le = LabelEncoder()
# le.fit(lables)
# lables2 = le.transform(lables)
# a = 1


