import numpy as np
import torch
import torch.nn.functional as F
import time
import tools
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
from model import Net

def pretrain(adj, features, para_set):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = adj.to(device)
    features = features.to(device)

    epochs = para_set['epoch2']
    lr = para_set['lr2']

    model_GCN = torch.load('model_autoencoder_GCN.pt')
    model_Net = Net(in_channels=para_set['mid_dim'],out_channels=para_set['out_dim'])
    model_Net = model_Net.to(device)
    print(model_GCN)
    print(model_Net)

    loss_func_GCN = torch.nn.CrossEntropyLoss().to(device)
    loss_func_Net = torch.nn.CrossEntropyLoss().to(device)
    optimizer_GCN = torch.optim.Adam(model_GCN.parameters(), lr)
    optimizer_Net = torch.optim.Adam(model_Net.parameters(), lr)
    Loss_list_GCN = []
    Loss_list_Net = []

    for epoch in range(1, epochs + 1):
        model_GCN.train()
        model_Net.train()
        optimizer_GCN.zero_grad() # 清空过往梯度
        optimizer_Net.zero_grad()
        z = model_GCN.encode(features, adj)
        #if epoch % 100 == 1:
        pse_label = tools.kmeans(features)
        pse_label = pse_label.to(device)
        loss_GCN = loss_func_GCN(z,pse_label)
        loss_GCN.backward(retain_graph=True)  # 梯度反向传播，保留梯度

        pre_label = model_Net(z)
        loss_Net = loss_func_Net(pre_label, pse_label)
        loss_Net.backward()  # 梯度反向传播
        # for name, parms in model.encoder.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,' -->grad_value:', parms.grad)
        cur_loss_GCN = loss_GCN.item()
        Loss_list_GCN.append(cur_loss_GCN)
        cur_loss_Net = loss_Net.item()
        Loss_list_Net.append(cur_loss_Net)
        optimizer_GCN.step()  # 优化网络参数
        optimizer_Net.step()
        print('pretrain: Epoch: {:4d}/{:4d} loss_GCN: {:.4f} loss_Net: {:.4f}'.format(epoch,epochs,cur_loss_GCN,cur_loss_Net))

    torch.save(model_GCN, 'model_pretrain_GCN.pt')
    torch.save(model_Net, 'model_pretrain_Net.pt')
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = Loss_list_GCN
    y2 = Loss_list_Net
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)  # plt.plot(x1, y1, 'o-')
    plt.title('loss_GCN vs. epoches')
    plt.ylabel('loss_GCN')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)  # plt.plot(x2, y2, '.-')
    plt.xlabel('loss_Net vs. epoches')
    plt.ylabel('loss_Net')
    plt.savefig("loss_pretrain.jpg")
    plt.close()
    #plt.show()

    print('------------pretrain Finished!  took %f seconds in total------------\n' % (time.time() - start_time))

def finetune(adj, features, real_label, para_set):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = adj.to(device)
    features = features.to(device)
    real_label_tensor = torch.tensor(real_label, dtype=torch.int64)
    real_label_tensor = real_label_tensor.to(device)

    epochs = para_set['epoch3']
    lr = para_set['lr3']
    Loss_list_GCN = []
    Loss_list_Net = []
    acc_list = []
    sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    for train_index, test_index in sfolder.split(features, real_label):
        model_GCN = torch.load('model_pretrain_GCN.pt')
        model_Net = torch.load('model_pretrain_Net.pt')
        print(model_GCN)
        print(model_Net)

        loss_func_GCN = torch.nn.CrossEntropyLoss().to(device)
        loss_func_Net = torch.nn.CrossEntropyLoss().to(device)
        optimizer_GCN = torch.optim.Adam(model_GCN.parameters(), lr)
        optimizer_Net = torch.optim.Adam(model_Net.parameters(), lr)

        for epoch in range(1, epochs + 1):
            model_GCN.train()
            model_Net.train()
            optimizer_GCN.zero_grad()  # 清空过往梯度
            optimizer_Net.zero_grad()

            z = model_GCN.encode(features, adj)
            loss_GCN = loss_func_GCN(z[train_index], real_label_tensor[train_index])
            loss_GCN.backward(retain_graph=True)  # 梯度反向传播

            pre_label = model_Net(z)
            loss_Net = loss_func_Net(pre_label[train_index], real_label_tensor[train_index])
            loss_Net.backward()  # 梯度反向传播
            cur_loss_GCN = loss_GCN.item()
            Loss_list_GCN.append(cur_loss_GCN)
            cur_loss_Net = loss_Net.item()
            Loss_list_Net.append(cur_loss_Net)
            optimizer_GCN.step()  # 优化网络参数
            optimizer_Net.step()
            #list(le.inverse_transform([2, 2, 1]))  # 逆过程
            print('finetune: Epoch: {:4d}/{:4d} loss_GCN: {:.4f} loss_Net: {:.4f}'.format(epoch,epochs,cur_loss_GCN,cur_loss_Net))

        model_GCN.eval()
        model_Net.eval()
        z = model_GCN.encode(features, adj)
        _, pre_label = model_Net(z).max(dim=1)

        corrects = float(pre_label[test_index].eq(real_label_tensor[test_index]).sum().item())
        acc = corrects / len(test_index)
        acc_list.append(acc)
        print('Accuracy:{:.4f}'.format(acc))

    print(acc_list)
    avg_acc = sum(acc_list) / len(acc_list)
    with open('record.txt', 'a') as f:
        f.write(str(avg_acc))
        f.write(str(para_set))
        f.write("\r\n")
    x1 = range(0, epochs*4)
    x2 = range(0, epochs*4)
    y1 = Loss_list_GCN
    y2 = Loss_list_Net
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)  # plt.plot(x1, y1, 'o-')
    plt.title('loss_GCN vs. epoches')
    plt.ylabel('loss_GCN')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)  # plt.plot(x2, y2, '.-')
    plt.xlabel('loss_Net vs. epoches')
    plt.ylabel('loss_Net')
    plt.savefig("loss_finetune.jpg")
    plt.close()
    #plt.show()

    print('------------finetune Finished!  took %f seconds in total------------\n' % (time.time() - start_time))