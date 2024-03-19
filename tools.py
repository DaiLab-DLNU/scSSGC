import pickle as pkl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import similarity
from fbpca import pca
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import faiss
def high_var_npdata(data, num, gene = None, ind=False): #data: cell * gene
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 0)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data[:, gene_ind]
    if ind:
        return data[:,gene_ind],gene_ind
    return data[:,gene_ind],gene.iloc[gene_ind]

def getgraph(matrix,dataset):

    matrix = matrix.values
    #n,m = matrix.shape
    n = len(matrix)
    matrix[np.eye(n,dtype=np.bool)] = 0
    adj = []
    adj_unique = []
    top_k = dataset['top_k']
    for i in range(len(matrix)):
        max_index = np.argpartition(matrix[i], -top_k)[-top_k:]  #np.argpartition()将传入的数组arr分成两部分，即：排在第k位置前面的数都小于k，排在第k位置后面的值都大于k。
        for j in range(top_k):
            if max_index[j]<i:
                ij_adj = [max_index[j],i]
            if max_index[j]>i:
                ij_adj = [i,max_index[j]]
            adj.append(ij_adj)
    for i in adj:
        if i not in adj_unique:
            adj_unique.append(i)
    adj_unique = torch.FloatTensor(adj_unique).t()
    adj_unique = adj_unique.type(torch.long)
    return adj_unique

def getgraph1(matrix,top_k):

    matrix = matrix.values
    #n,m = matrix.shape
    n = len(matrix)
    matrix[np.eye(n,dtype=np.bool)] = 0
    adj = []
    adj_unique = []
    for i in range(len(matrix)):
        min_index = np.argpartition(matrix[i], -top_k)[0:-top_k]  #np.argpartition()将传入的数组arr分成两部分，即：排在第k位置前面的数都小于k，排在第k位置后面的值都大于k。
        matrix[i][min_index] = 0
    adj = sp.csr_matrix(matrix)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def getgraph2(matrix,top_k):

    matrix = matrix.values
    #n,m = matrix.shape
    n = len(matrix)
    matrix[np.eye(n,dtype=np.bool)] = 0
    adj = []
    adj_unique = []
    for i in range(len(matrix)):
        max_index = np.argpartition(matrix[i], -top_k)[-top_k:]  #np.argpartition()将传入的数组arr分成两部分，即：排在第k位置前面的数都小于k，排在第k位置后面的值都大于k。
        for j in range(top_k):
            if max_index[j]<i:
                ij_adj = [max_index[j],i]
            if max_index[j]>i:
                ij_adj = [i,max_index[j]]
            adj.append(ij_adj)
    for i in adj:
        if i not in adj_unique:
            adj_unique.append(i)
    adj_unique = torch.FloatTensor(adj_unique).t()
    adj_unique = adj_unique.type(torch.long)
    return adj_unique

def getgraph3(data,top_k):
    adj = []
    adj_unique = []
    print('start ')
    index =nearest_neighbor_search_faiss(data, top_k)
    print('end')
    #index = nearest_neighbor_search(data, dataset)
    for i in range(len(index)):#for i in trange(len(index)):
        for j in range(top_k):
            if index[i][j]<i:
                ij_adj = [index[i][j],i]

                adj.append(ij_adj)
            elif index[i][j]>i:
                ij_adj = [i,index[i][j]]

                adj.append(ij_adj)
            else:
                pass
        print('{}'.format(i))
    adj = np.unique(adj, axis=0)
    # for i in adj:
    #     print(i)
    #     if i not in adj_unique:
    #         adj_unique.append(i)
    adj = torch.FloatTensor(adj).t()
    print('123')
    adj = adj.type(torch.long)
    print('123')
    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def features_normalize(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

def getmatrix(data,para_set):

    # similarity mitrix
    sim = similarity.similarity(para_set['nearest_k'])
    matrix = sim.sim_matrix(data)
    matrix = torch.tensor(matrix)
    matrix = torch.tanh(matrix)
    matrix = np.array(matrix)
    matrix = pd.DataFrame(matrix)
    return matrix

def kmeans(z):
    z_copy = z.cpu()
    z_copy = z_copy.detach().numpy()
    kmeans = KMeans(max_iter = 500).fit(z_copy)
    #print(kmeans.labels_)
    label = kmeans.labels_
    pse_label = torch.tensor(label,dtype=torch.int64)

    return pse_label

def clean_10(data,label):
    type_list = np.unique(label)
    n_type = len(type_list)
    type_cellnum = np.zeros(n_type)
    index = []
    for i in range(n_type):
        index_i = []
        for j in range(len(label)):
            if label[j] == type_list[i]:
                type_cellnum[i] += 1
                index_i.append(j)
        index.append(index_i)
    clean_index = []
    for i in range(len(type_cellnum)):
        if type_cellnum[i]>10:
            clean_index.extend(index[i])
    data = data[clean_index]
    label = label[clean_index]
    print('after cleaned,the cell num is: {:4d}'.format(len(clean_index)))
    return data,label

def get_pretrain_label(feature):
    avg_cluster_num = 100
    feature = feature.astype('float32')
    label1 = run_kmeans(feature, len(feature) // avg_cluster_num)
    label2 = run_kmeans(feature, len(feature) // (avg_cluster_num // 2))
    label3 = run_kmeans(feature, len(feature) // (avg_cluster_num * 2) + 1)
    pse_label = np.stack((label1, label2, label3), axis=-1)

    return pse_label

def get_augmented_features(concat,features,device,cvae_model):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features).detach()
        # augmented_features = torch.randn([2119, 18915]).to(device)
        X_list.append(augmented_features)
    return X_list

def get_pretrain_loader(feature,adj,concat,emb=None,cvae_model=None):
    label_num_list = []
    pse_label_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if emb is None:
        pse_label = get_pretrain_label(feature)
    else:
        emb.to(device)
        emb.eval()
        X_list = get_augmented_features(concat,feature,device,cvae_model)
        feature_out = emb(X_list+[torch.FloatTensor(feature).to(device)],adj)
        feature_out = feature_out.cpu()
        feature_out = feature_out.detach().numpy()
        pse_label = get_pretrain_label(feature_out)

    label_num_list.append(pse_label.max(axis=0) + 1)
    pse_label_list.append(torch.LongTensor(pse_label).to(device))
    return label_num_list,pse_label_list

def run_kmeans(x, nmb_clusters):
    x = np.ascontiguousarray(x)
    n_data, d = x.shape
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    index=faiss.IndexFlatL2(d)
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

def nearest_neighbor_search_faiss(data,top_k):
    start_time = time.time()
    n ,d = data.shape
    data = np.ascontiguousarray(data)
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(data)
    index = np.zeros((n, top_k))
    for i in range(n):
        D, I = gpu_index_flat.search(data[i].reshape(1,-1), top_k)
        index[i, :] = I
        print("\r#1 get nearest neighbors------------{:.2f}%------------".format(100 * i / n), end='')
    print('\r------------get nearest neighbors!  took %f seconds in total------------\n' % (time.time() - start_time))
    return index.astype('int')#, value