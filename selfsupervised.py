import numpy as np
import torch
import torch.nn.functional as F
import time
import tools
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
from model import Net
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from torch_geometric.nn import GAE
import torch.nn as nn
from model import GCNEncoder,mtclf,LASAGE,clf
from torch_geometric.nn import GATConv
from tqdm import trange
import copy
from torch_geometric.nn import SAGEConv
import random
from torch.nn import Parameter
import os

def pretrain(adj, features, para_set,i,first):
    start_time = time.time()
    n_nodes, feat_dim = features.shape
    epochs = para_set['epoch2']
    lr = para_set['lr2']
    hidden = para_set['hidden']
    hidden2 = para_set['hidden2']
    dropout = para_set['dropout']
    num_layers = para_set['num_layers']
    lam = para_set['lam']
    concat = para_set['concat']
    samples= para_set['samples']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae_model = torch.load(para_set['dataname']+'cvae.pkl', map_location=device)
    adj = adj.to(device)
    if first:
        emb = None
        label_num_list,pse_label = tools.get_pretrain_loader(features,adj,concat,emb)
    else:
        with open(para_set['dataname']+'Pretrain.out', 'rb') as f:
            emb=torch.load(f)
        enc = LASAGE(concat+1, feat_dim, hidden, hidden2,num_layers,dropout)
        enc.load_state_dict(emb)
        label_num_list,pse_label = tools.get_pretrain_loader(features,adj,concat,enc,cvae_model)


    if first:
        model = mtclf(concat+1,feat_dim, hidden,hidden2, label_num_list,num_layers,dropout)
    else:
        model = mtclf(concat+1,feat_dim, hidden,hidden2, label_num_list,num_layers,dropout, enc)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    features_normalized = tools.features_normalize(features)
    features_normalized = torch.FloatTensor(features)



    X_list = tools.get_augmented_features(concat,features,device,cvae_model)

    features_normalized = features_normalized.to(device)
    cnt = 5
    min_loss = 1000
    pt_patience = 5
    for e in range(epochs):
        total = 0
        model.train()
        for k in range(samples):
            X_list = tools.get_augmented_features(concat,features,device,cvae_model)
        feature_list = []
        feature_list.append(X_list + [features_normalized])
        loss = model(feature_list,adj, pse_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = loss
        print(e, total)
        if total < min_loss:
            min_loss = total
            cnt = pt_patience
        else:
            cnt -= 1
            if cnt == 0: break

    print('------------pretrain Finished!  took %f seconds in total------------\n' % (time.time() - start_time))
    with open(para_set['dataname']+'Pretrain.out', 'wb') as f:
        torch.save(model.encoder.state_dict(), f)

def finetune_with_pretrain(adj, features, real_label,num_label, para_set,i):

    start_time = time.time()
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    hidden = para_set['hidden']
    dropout = para_set['dropout']
    num_layers = para_set['num_layers']
    lam = para_set['lam']
    concat = para_set['concat']
    samples = para_set['samples']
    epochs = para_set['epoch3']
    lr = para_set['lr3']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = adj.to(device)
    features_normalized = tools.features_normalize(features)
    features_normalized = torch.FloatTensor(features)

    cvae_model = torch.load(para_set['dataname']+'cvae.pkl', map_location=device)

    X_list = tools.get_augmented_features(concat,features,device,cvae_model)

    real_label_tensor = torch.tensor(real_label, dtype=torch.int64)
    real_label_tensor = real_label_tensor.to(device)
    features_normalized = features_normalized.to(device)


    Train_Loss_list_1 = []
    val_acc_list_1 = []
    y_pred = np.array([])
    y_true = np.array([])
    j = 0

    hidden2 = para_set['hidden2']
    criterion = torch.nn.BCEWithLogitsLoss()

    shuffle_index = np.random.permutation(features.shape[0])
    # np.savetxt(para_set['dataset'] + '/shuffle_index_' + '.txt')
    train_size, val_size = int(len(shuffle_index) * 0.8), int(len(shuffle_index) * 0.9)
    train_index = shuffle_index[0:train_size]
    val_index = shuffle_index[train_size:val_size]
    test_index = shuffle_index[val_size:]

    # Model and optimizer
    with open(para_set['dataname'] + 'Pretrain.out', 'rb') as f:
        pretrain_model = torch.load(f)
        embed_ft = LASAGE(concat + 1, features.shape[1], hidden, hidden2, num_layers, dropout)
        embed_ft.load_state_dict(pretrain_model)
    model = clf(concat + 1, features.shape[1], hidden, hidden2, num_label, num_layers, dropout, embed_ft).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    # Train model
    cnt = 50
    min_loss = 100
    pt_patience = 50
    for epoch in range(1, epochs + 1):
        model.train()
        output_list = []
        #for k in range(samples):
        X_list = tools.get_augmented_features(concat, features, device, cvae_model)
        out = model(X_list + [features_normalized], adj)
        #output_list.append(out)

        loss_train = F.cross_entropy(out[train_index], real_label_tensor[train_index])
        Train_Loss_list_1.append(loss_train.item())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total = float(loss_train)

        with torch.no_grad():
            val_acc = 0
            val_put = torch.log_softmax(out, dim=1)
            _, pre_label = val_put.max(dim=1)
            val_label_copy = pre_label.cpu()
            val_label_copy = val_label_copy.detach().numpy()
            val_pred = val_label_copy[val_index]
            val_true = real_label[val_index]
            val_acc = metrics.accuracy_score(val_pred, val_true)
            val_acc_list_1.append(val_acc)
        print('{} {:2d} {:2d}  finetune: Epoch: {:4d}/{:4d} loss: {:.4f} val acc:{:.4f}'.format(para_set['dataname'], i, j, epoch, epochs, loss_train.item(),val_acc))


        # if total<min_loss:
        #     min_loss = total
        #     cnt = pt_patience
        # else:
        #     cnt-=1
        #     if cnt==0:break
    with torch.no_grad():
        model.eval()
        X_list = tools.get_augmented_features(concat, features, device, cvae_model)
        output = model(X_list + [features_normalized], adj)
        output = torch.log_softmax(output, dim=1)
        _, pre_label = output.max(dim=1)
        pre_label_copy = pre_label.cpu()
        pre_label_copy = pre_label_copy.detach().numpy()
        y_pred = pre_label_copy[test_index]
        y_true = real_label[test_index]

    test_acc = metrics.accuracy_score(y_pred, y_true)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=list(range(num_label)), average=None)
    # print(np.mean(p_class))
    # print(np.median(p_class))
    print('test_acc:{:.4f}'.format(test_acc))
    print(f_class)
    mid_f1 = np.median(f_class)
    print('mid_f1:{:.4f}'.format(mid_f1))
    print('------------finetune Finished!  took %f seconds in total------------\n' % (time.time() - start_time))
    os.remove(para_set['dataname'] + 'Pretrain.out')
    #os.remove(para_set['dataname'] + 'cvae.pkl')
    return test_acc, mid_f1,Train_Loss_list_1,val_acc_list_1

def finetune_without_pretrain(adj, features, real_label,num_label, para_set,i,num_g):

    start_time = time.time()
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    hidden = para_set['hidden']
    dropout = para_set['dropout']
    num_layers = para_set['num_layers']
    lam = para_set['lam']
    concat = para_set['concat']
    samples = para_set['samples']
    epochs = para_set['epoch3']
    lr = para_set['lr3']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = adj.to(device)
    features_normalized = tools.features_normalize(features)
    features_normalized = torch.FloatTensor(features)

    cvae_model = torch.load(para_set['dataname']+'cvae.pkl', map_location=device)

    X_list = tools.get_augmented_features(concat,features,device,cvae_model)

    real_label_tensor = torch.tensor(real_label, dtype=torch.int64)
    real_label_tensor = real_label_tensor.to(device)
    features_normalized = features_normalized.to(device)


    Train_Loss_list_2 = []
    val_acc_list_2 = []
    y_pred = np.array([])
    y_true = np.array([])
    j = 0

    hidden2 = para_set['hidden2']
    criterion = torch.nn.BCEWithLogitsLoss()

    shuffle_index = np.random.permutation(features.shape[0])
    # np.savetxt(para_set['dataset'] + '/shuffle_index_' + '.txt')
    train_size, val_size = int(len(shuffle_index) * 0.8), int(len(shuffle_index) * 0.9)
    train_index = shuffle_index[0:train_size]
    val_index = shuffle_index[train_size:val_size]
    test_index = shuffle_index[val_size:]

    # Model and optimizer
    # with open(para_set['dataname'] + 'Pretrain.out', 'rb') as f:
    #     pretrain_model = torch.load(f)
    #     embed_ft = LASAGE(concat + 1, features.shape[1], hidden, hidden2, num_layers, dropout)
    #     embed_ft.load_state_dict(pretrain_model)
    model = clf(concat + 1, features.shape[1], hidden, hidden2, num_label, num_layers, dropout).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    # Train model
    cnt = 50
    min_loss = 100
    pt_patience = 50
    for epoch in range(1, epochs + 1):
        model.train()
        output_list = []
        #for k in range(samples):
        X_list = tools.get_augmented_features(concat, features, device, cvae_model)
        out = model(X_list + [features_normalized], adj)
        #output_list.append(out)

        loss_train = F.cross_entropy(out[train_index], real_label_tensor[train_index])
        Train_Loss_list_2.append(loss_train.item())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total = float(loss_train)

        with torch.no_grad():
            val_acc = 0
            val_put = torch.log_softmax(out, dim=1)
            _, pre_label = val_put.max(dim=1)
            val_label_copy = pre_label.cpu()
            val_label_copy = val_label_copy.detach().numpy()
            val_pred = val_label_copy[val_index]
            val_true = real_label[val_index]
            val_acc = metrics.accuracy_score(val_pred, val_true)
            val_acc_list_2.append(val_acc)
        print('{} {:2d} {:2d}  finetune: Epoch: {:4d}/{:4d} loss: {:.4f} val acc:{:.4f}'.format(para_set['dataname'], i, j, epoch, epochs, loss_train.item(),val_acc))

        # if total<min_loss:
        #     min_loss = total
        #     cnt = pt_patience
        # else:
        #     cnt-=1
        #     if cnt==0:break
    with torch.no_grad():
        model.eval()
        X_list = tools.get_augmented_features(concat, features, device, cvae_model)
        output = model(X_list + [features_normalized], adj)
        output = torch.log_softmax(output, dim=1)
        _, pre_label = output.max(dim=1)
        pre_label_copy = pre_label.cpu()
        pre_label_copy = pre_label_copy.detach().numpy()
        y_pred = pre_label_copy[test_index]
        y_true = real_label[test_index]
    labels = np.unique(y_true)
    test_acc = metrics.accuracy_score(y_pred, y_true)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels, average=None)
    # print(np.mean(p_class))
    # print(np.median(p_class))
    print('test_acc:{:.4f}'.format(test_acc))
    print(f_class)
    mid_f1 = np.median(f_class)
    print('mid_f1:{:.4f}'.format(mid_f1))
    print('------------finetune Finished!  took %f seconds in total------------\n' % (time.time() - start_time))
    #os.remove(para_set['dataname'] + 'Pretrain.out')
    os.remove(para_set['dataname'] + 'cvae.pkl')
    return test_acc, mid_f1,Train_Loss_list_2,val_acc_list_2