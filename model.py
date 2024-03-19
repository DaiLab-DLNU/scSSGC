from torch_geometric.nn import SAGEConv#GCNConv, SAGEConv, GATConv
import torch
import torch.nn.functional as F
import numpy as np

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = SAGEConv(in_channels, 2 * out_channels) # cached only for transductive learning
        self.conv2 = SAGEConv(2 * out_channels, out_channels) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    # def forward(self, x, edge_index):
    #     x = self.conv1(x, edge_index).relu()
    #     x = self.conv2(x, edge_index)
    #     output = F.softmax(x, dim=1)
    #     return output

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        output = F.log_softmax(self.l1(x), dim=1)
        return output

import torch.nn as nn


class LASAGE(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LASAGE, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        for _ in range(concat):
            self.convs_initial.append(SAGEConv(in_channels, hidden_channels))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(concat * hidden_channels, concat * hidden_channels))
        self.convs.append(SAGEConv(concat * hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()

    def forward(self, x_list, adj_t):
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(x_list[i], adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0):

        super().__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size)

    def forward(self, x, c=None):

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class mtclf(nn.Module):
    def __init__(self,concat, feat_dim, hid,out_channels, num_cl,num_layers,dropout, pt_encoder=None):
        super().__init__()
        self.len = len(num_cl)
        self.num_cl = np.array(num_cl)
        if pt_encoder is None:
            # if config.pca_pt:
            #     self.encoder = encoder(config.pca_dim, hid)
            # else:
            self.encoder = LASAGE(concat, feat_dim, hid, out_channels, num_layers,dropout)
        else:
            self.encoder = pt_encoder
            self.encoder.train()
        self.mt = nn.ModuleList([SAGEConv(out_channels, n) for n in self.num_cl[:, 0]])
        self.mt2 = nn.ModuleList([SAGEConv(out_channels, n) for n in self.num_cl[:, 1]])
        self.mt3 = nn.ModuleList([SAGEConv(out_channels, n) for n in self.num_cl[:, 2]])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, adj, y):
        h = []
        out, out1, out2 = [], [], []

        for k, v in enumerate(range(self.len)):
            h.append(torch.sigmoid(self.encoder(X[k],adj)))
            out.append(torch.sigmoid(self.mt[k](h[k],adj)))
            out1.append(torch.sigmoid(self.mt2[k](h[k],adj)))
            out2.append(torch.sigmoid(self.mt3[k](h[k],adj)))
            if k == 0:
                l = self.loss(out[k], y[k][:, 0]) + self.loss(out1[k], y[k][:, 1]) + self.loss(out2[k], y[k][:, 2])
            else:
                l = l + self.loss(out[k], y[k][:, 0]) + self.loss(out1[k], y[k][:, 1]) + self.loss(out2[k], y[k][:, 2])
        return l


# class encoder(nn.Module):
#     def __init__(self, num_g, hid):
#         super().__init__()
#         self.fc1 = nn.Linear(num_g, 200)
#         # self.fc2=nn.Linear(200,hid)
#
#     def forward(self, X):
#         # h=torch.sigmoid(self.fc1(X))
#         return self.fc1(X)

class clf(nn.Module):
    def __init__(self,concat, num_g, hid,out_channels, num_l,num_layers,dropout, pt_encoder=None):
        super().__init__()
        if pt_encoder is None:
            # if config.pca_ft:
            #     self.encoder=encoder(config.pca_dim,hid)
            # else:
            self.encoder = LASAGE(concat, num_g, hid, out_channels,num_layers,dropout)
        else:
            self.encoder=pt_encoder
        self.fc=SAGEConv(out_channels,num_l)
        self.dropout = dropout
    def forward(self, X, adj):
        h = F.relu(self.encoder(X,adj))
        #h = F.dropout(h, p=self.dropout, training=self.training)
        out=self.fc(h,adj)
        return out
