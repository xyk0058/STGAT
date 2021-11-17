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

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Too harsh to use the same dropout. TODO add another dropout
        # input = F.dropout(input, self.dropout, training=self.training)

        seq = torch.transpose(input, 0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq)

        f_1 = self.f_1(seq_fts)
        f_2 = self.f_2(seq_fts)
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
        coefs = F.dropout(coefs, self.dropout, training=self.training)

        ret = torch.mm(coefs, seq_fts)# + self.bias

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
            else:
                ret += input

        if self.concat:
            return F.elu(ret)
        else:
            return ret

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
        self.attentions = [GraphAttentionLayer(out_channels*(num_timesteps_input), spatial_channels, 
                            dropout=dropout, alpha=alpha, concat=concat) for _ in range(nheads)]
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        if cuda:
            self.attentions = [att.cuda() for att in self.attentions]
            # self.attentions2 = [att.cuda() for att in self.attentions2]
    
    def forward(self, X, A_hat):
        residual = X
        t = self.temporal1(X)
        t = t.contiguous().view(t.shape[0], t.shape[1], -1)
        if self.concat:
            t2 = torch.cat([att(t, A_hat) for att in self.attentions], dim=2)
        else:
            t2 = sum([att(t, A_hat) for att in self.attentions]) / self.nheads

        t2 = t2.view(t2.shape[0], t2.shape[1], -1, self.spatial_channels)

        # t3 = self.temporal2(t2)
        t3 = t2
        # print('residual', t3.shape, residual.shape)
        if t3.shape[-1] == residual.shape[-1]:
            t3 = self.relu(t3 + residual[:,:,-t3.shape[2]:,:])
        else:
            t3 = self.relu(t3)
        return self.batch_norm(t3)



class STGAT(nn.Module):
    def __init__(self, cuda, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, nheads=6, nhid=64, layers=4, end_channels=512):
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
        # self.last_temporal = TimeBlock(in_channels=nhid, out_channels=nhid, cuda=cuda)

        # self.output = nn.Sequential(
        #     nn.Linear(nhid * nheads, num_timesteps_output),
        #     nn.ReLU()
        # )
        
        self.output = nn.Sequential(
            nn.Conv1d(nhid, end_channels, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(end_channels),
            nn.Conv1d(end_channels, num_timesteps_output, 1, padding=0)
        )
        
        self.time_decay_mult = nn.Parameter(torch.ones(1) * -0.1)
        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(nhid)

        # self.A_hat = nn.Parameter(torch.ones(num_nodes, num_nodes))
    
    def forward(self, A_hat, X, A_hat_=None, X_=None):
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
        # print('out3', out.shape)
        # out3 = self.last_temporal(out)
        emb = out
        emb = emb.reshape((emb.shape[0], emb.shape[1], -1)).contiguous()

        logits = None
        # if self.training:
        #     c = self.sigm(self.readout(emb, None))
        #     out_ = X_
        #     for i in range(self.layers):
        #         out_ = self.blocks[i](out_, A_hat)
        #     # out3_ = self.last_temporal(out_)
        #     emb_ = out_
        #     # emb_ = self.bn(out3_)
        #     emb_ = emb_.reshape((emb_.shape[0], emb_.shape[1], -1)).contiguous()
        #     logits = self.disc(c, emb, emb_, None, None)
        
        
        emb = emb.permute(0, 2, 1).contiguous()
        out4 = self.output(emb).permute(0, 2, 1).unsqueeze(dim=3)

        if self.training:
            return out4, logits
        else:
            return out4
