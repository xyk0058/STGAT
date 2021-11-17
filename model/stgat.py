import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .layers import TimeBlock
from .readout import AvgReadout
from .discriminator import Discriminator



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_nodes, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2.0))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a2.data, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.downsample = nn.Conv1d(in_features, out_features, 1)

        self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))


    def forward(self, input, adj):
        batch_size = input.size(0)
        h = torch.bmm(input, self.W.expand(batch_size, self.in_features, self.out_features))
        f_1 = torch.bmm(h, self.a1.expand(batch_size, self.out_features, 1))
        f_2 = torch.bmm(h, self.a2.expand(batch_size, self.out_features, 1))
        e = self.leakyrelu(f_1 + f_2.transpose(2,1))
        # add by xyk
        attention = torch.mul(adj, e)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h) + self.bias.expand(batch_size, self.num_nodes, self.out_features)
        if input.shape[-1] != h_prime.shape[-1]:
            input = self.downsample(input.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            h_prime = h_prime + input
        else:
            h_prime = h_prime + input
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class STGATBlock(nn.Module):
    def __init__(self, cuda, in_channels, spatial_channels, out_channels,
                num_nodes, num_timesteps_input, dropout=0.6, alpha=0.2, nheads=6, concat=True):
        super(STGATBlock, self).__init__()
        self.nheads = nheads
        self.concat = concat
        self.cuda = cuda
        self.spatial_channels = spatial_channels
        self.temporal1 = nn.Sequential(TimeBlock(in_channels=in_channels, cuda=cuda,
                                   out_channels=out_channels),
        )
        self.attentions = [GraphAttentionLayer(out_channels*(num_timesteps_input), spatial_channels, num_nodes=num_nodes,
                            dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        if cuda:
            self.attentions = [att.cuda() for att in self.attentions]
    
    def forward(self, X, A_hat):
        residual = X
        t = self.temporal1(X)
        t = t.contiguous().view(t.shape[0], t.shape[1], -1)
        if self.concat:
            t2 = torch.cat([att(t, A_hat) for att in self.attentions], dim=2)
        else:
            t2 = sum([att(t, A_hat) for att in self.attentions]) / self.nheads

        t2 = t2.view(t2.shape[0], t2.shape[1], -1, self.spatial_channels)
        t3 = t2
        if t3.shape[-1] == residual.shape[-1]:
            t3 = t3 + residual[:,:,-t3.shape[2]:,:]
        else:
            t3 = t3
        return self.relu(self.batch_norm(t3))




class GatedLinearUnits(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=16, kernel_size=2, dilation=1, groups=4, activate=False):
        super(GatedLinearUnits, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activate = activate

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True, groups=groups)
        nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.conv.bias, 0.1)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True, groups=groups)
        nn.init.xavier_uniform_(self.gate.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.gate.bias, 0.1)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.downsample.bias, 0.1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

        self.sigmod = nn.Sigmoid()
        
    def forward(self, X):
        res = X
        gate = X
        
        X = nn.functional.pad(X, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        out = self.conv(X)
        if self.activate:
            out = F.tanh(out)

        gate = nn.functional.pad(gate, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        gate = self.gate(gate)
        gate = self.sigmod(gate)

        out = torch.mul(out, gate)
        ones = torch.ones_like(gate)

        if res.shape[1] != out.shape[1]:
            res = self.downsample(res)
        
        res = torch.mul(res, ones-gate)
        out = out + res
        out = self.relu(self.bn(out))
        return out


class EndConv(nn.Module):
    def __init__(self, in_channels, out_channels, nhid_channels, layer=3):
        super(EndConv, self).__init__()
        layers = []
        for i in range(layer):
            # print('in_channels', in_channels)
            if i == 0:
                layers.append(GatedLinearUnits(in_channels, nhid_channels, kernel_size=1, dilation=1, groups=1))
            else:
                layers.append(GatedLinearUnits(nhid_channels, nhid_channels, kernel_size=3, dilation=1, groups=1))
        layers.append(nn.Conv1d(nhid_channels, out_channels, 1))
        self.units = nn.Sequential(*layers)
    
    def forward(self, X):
        out = self.units(X)
        return out


class STGAT(nn.Module):
    def __init__(self, cuda, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nheads=4, nhid=64, layers=4):
        super(STGAT, self).__init__()
        self.cuda_device = cuda
        self.num_timesteps_output = num_timesteps_output
        self.nheads = nheads
        self.layers = layers
        self.blocks = nn.ModuleList()
        for i in range(layers):
            in_channels = nhid
            n_input = nheads
            concat = True
            if i == 0:
                in_channels = num_features
                n_input = num_timesteps_input
            if i == layers-1:
                concat = False
                nheads += 2
            self.blocks.append( STGATBlock(cuda, in_channels=in_channels, out_channels=nhid, concat=concat,
                                 spatial_channels=nhid, num_nodes=num_nodes, num_timesteps_input=n_input, nheads=nheads)
                            )
        self.output = EndConv(nhid, num_timesteps_output, 512)
        # self.output = nn.Sequential(
        #     nn.Conv1d(nhid, 512, 1, dilation=1, padding=0),
        #     nn.BatchNorm1d(512, momentum=0.2),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 512, 1, dilation=1, padding=0),
        #     nn.BatchNorm1d(512, momentum=0.2),
        #     nn.ReLU(),
        #     nn.Conv1d(512, num_timesteps_output, 1, dilation=1, padding=0),
        #     # nn.BatchNorm1d(num_timesteps_output, momentum=0.2),
        # )
        # self.output = nn.Conv1d(nheads * 64, num_timesteps_output, 3, padding=1)

        # self.readout = AvgReadout()
        # self.sigm = nn.Sigmoid()
        # self.disc = Discriminator(nheads * 64)
        self.time_decay_mult = nn.Parameter(torch.ones(1) * -0.1)
        


    
    def forward(self, A_hat, X, A_hat_=None, X_=None):
        # out1 = self.block1(X, A_hat)
        # out2 = self.block2(out1, A_hat)

        # time_decay
        # time_decay = torch.zeros(12, 2)
        # for i in range(12):
        #     time_decay[11-i] = torch.exp(self.time_decay_mult * i)
        # if self.cuda_device:
        #     time_decay = time_decay.cuda()
        # X = X * time_decay
        out = X
        for i in range(self.layers):
            out = self.blocks[i](out, A_hat)
        # out3 = self.last_temporal(out)
        # emb = self.bn(out3)
        emb = out
        emb = emb.reshape((emb.shape[0], emb.shape[1], -1)).permute(0, 2, 1)

        logits = None
        # if self.training:
        #     c = self.sigm(self.readout(emb, None))
        #     out1_ = self.block1(X_, A_hat_)
        #     out2_ = self.block2(out1_, A_hat_)
        #     # out2_ = self.block3(out2_, A_hat_)
        #     out3_ = self.last_temporal(out2_)
        #     emb_ = self.bn(out3_)
        #     emb_ = emb_.reshape((emb_.shape[0], emb_.shape[1], -1))
        #     logits = self.disc(c, emb, emb_, None, None)
        # print(emb.shape)
        out4 = self.output(emb).permute(0, 2, 1).unsqueeze(dim=3)
        # time_decay
        # time_decay = torch.zeros(12, 1)
        # for i in range(12):
        #     time_decay[i] = torch.exp(self.time_decay_mult * i)
        # if self.cuda_device:
        #     time_decay = time_decay.cuda()
        # out4 = out4 * time_decay
        # print('out4', out4.shape)
        if self.training:
            return out4, logits
        else:
            return out4
