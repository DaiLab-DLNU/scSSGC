import torch
import numpy as np
import time
from torch_geometric.nn import GAE
import scipy.sparse as sp
from model import GCNEncoder
import matplotlib.pyplot as plt
from model import VAE
from tqdm import tqdm, trange
import gc
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = (BCE + KLD) / x.size(0)
    print(loss)
    return loss

def GraphAutoEncoder(adj, features, para_set,is_large):
    start_time = time.time()
    n_nodes, feat_dim = features.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = para_set['epoch1']
    lr = para_set['lr1']
    latent_size = 32
    conditional = 'True'

    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256],
               latent_size=latent_size,
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=conditional,
               conditional_size=features.shape[1])
    cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)
    cvae.to(device)

    x_list, c_list = [], []
    if not is_large:
        for i in trange(adj.shape[0]):
            x = features[adj[i].nonzero()[1]]
            c = np.tile(features[i], (x.shape[0], 1))
            x_list.append(x)
            c_list.append(c)
        features_x = np.vstack(x_list)
        features_c = np.vstack(c_list)
    else:
        for i in trange(adj.shape[1]):
            x = features[adj[0][i]]
            # print(x.shape[0])
            c = features[adj[1][i]]
            x_list.append(x)
            c_list.append(c)
        features_x = np.vstack(x_list)
        features_c = np.vstack(c_list)
    del x_list
    del c_list
    gc.collect()
    batch_size = 256
    for epoch in trange(epochs, desc='Run CVAE Train'):
        index = random.sample(range(features_c.shape[0]), batch_size)
        x, c = features_x[index], features_c[index]
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        cvae.train()
        x, c = x.to(device), c.to(device)
        if conditional:
            recon_x, mean, log_var, _ = cvae(x, c)
        else:
            recon_x, mean, log_var, _ = cvae(x)
        cvae_loss = loss_fn(recon_x, x, mean, log_var)
        cvae_optimizer.zero_grad()
        cvae_loss.backward()
        cvae_optimizer.step()

    del (features_x)
    del (features_c)
    gc.collect()

    torch.save(cvae, para_set['dataname'] + 'cvae.pkl')
