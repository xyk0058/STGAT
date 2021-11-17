import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class GatedLinearUnits(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=16, kernel_size=2, dilation=1, cuda=True, groups=4, activate=False):
        super(GatedLinearUnits, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cuda = cuda
        self.activate = activate

        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation), bias=True, groups=groups))
        nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.conv.bias, 0.1)
        self.gate = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation), bias=True, groups=groups))
        nn.init.xavier_uniform_(self.gate.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.gate.bias, 0.1)
        self.downsample = weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1), bias=True))
        nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.downsample.bias, 0.1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.2)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0.1)

        self.sigmod = nn.Sigmoid()
        
    def forward(self, X):
        res = X
        gate = X
        print('X_in', X.shape)
        X = nn.functional.pad(X, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        out = self.conv(X)
        if self.activate:
            out = F.tanh(out)

        gate = nn.functional.pad(gate, ((self.kernel_size-1)*self.dilation, 0, 0, 0))
        gate = self.gate(gate)
        gate = self.sigmod(gate)

        out = torch.mul(out, gate)
        ones = torch.ones_like(gate)

        # print('res', res.shape, out.shape)
        if res.shape[1] != out.shape[1]:
            res = self.downsample(res)
        res = torch.mul(res, ones-gate)
        out = out + res
        # out = self.bn(self.relu(out))
        out = self.relu(self.bn(out))
        print('X_out', out.shape)
        return out


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, nhid_channels=128, dropout=0.6, layer=3, cuda=True):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        layers = []
        for i in range(layer):
            # print('in_channels', in_channels)
            if i == 0:
                layers.append(GatedLinearUnits(in_channels, nhid_channels, kernel_size=1, dilation=1, cuda=cuda, groups=1))
            elif i == layer-1:
                layers.append(GatedLinearUnits(nhid_channels, out_channels, kernel_size=1, dilation=1, cuda=cuda, groups=1))
            else:
                layers.append(GatedLinearUnits(nhid_channels, nhid_channels, kernel_size=kernel_size, dilation=2**(i), cuda=cuda, groups=1))
        # layers = [GatedLinearUnits(in_channels, out_channels, kernel_size=kernel_size, dilation=2**i, cuda=cuda) for i in range(layer)]
        self.units = nn.Sequential(*layers)

    def forward(self, X):
        # print('in', X.shape)
        X = X.permute(0, 3, 1, 2)
        out = self.units(X)
        out = out.permute(0, 2, 3, 1)
        # print('out', X.shape)
        return out

# class TimeBlock(nn.Module):
#     """
#     Neural network block that applies a temporal convolution to each node of
#     a graph in isolation.
#     """

#     def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.6, dilation=1):
#         """
#         :param in_channels: Number of input features at each node in each time
#         step.
#         :param out_channels: Desired number of output channels at each node in
#         each time step.
#         :param kernel_size: Size of the 1D temporal kernel.
#         """
#         super(TimeBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, dilation), dilation=dilation))
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.BatchNorm2d(out_channels, momentum=0.2)
#         # self.conv2 = weight_norm(nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, dilation), dilation=dilation))
#         # self.relu2 = nn.ReLU()
#         # self.dropout2 = nn.BatchNorm2d(out_channels, momentum=0.2)
#         self.gate = weight_norm(nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, dilation), dilation=dilation))
#         self.downsample = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 1)))
#         self.sigmod = nn.Tanh()
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         # self.conv2.weight.data.normal_(0, 0.01)
#         self.gate.weight.data.normal_(0, 0.01)
#         # self.downsample.weight.normal_(0, 0.01)
#         torch.nn.init.normal(self.downsample.weight, 0, 0.01)


#     def forward(self, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels)
#         :return: Output data of shape (batch_size, num_nodes,
#         num_timesteps_out, num_features_out=out_channels)
#         """
#         X = X.permute(0, 3, 1, 2)
#         out1 = self.dropout1(self.relu1(self.conv1(X)))

#         gate = self.sigmod(self.gate(out1))
#         out = out1 * gate

#         res = self.downsample(X)
#         out = self.relu(out + res)

#         out = out.permute(0, 2, 3, 1)
#         return out

# class TimeBlock(nn.Module):
#     """
#     Neural network block that applies a temporal convolution to each node of
#     a graph in isolation.
#     """

#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         """
#         :param in_channels: Number of input features at each node in each time
#         step.
#         :param out_channels: Desired number of output channels at each node in
#         each time step.
#         :param kernel_size: Size of the 1D temporal kernel.
#         """
#         super(TimeBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 1), dilation=1))
#         self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 2), dilation=2)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 4), dilation=4)

#     def forward(self, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels)
#         :return: Output data of shape (batch_size, num_nodes,
#         num_timesteps_out, num_features_out=out_channels)
#         """
#         X = X.permute(0, 3, 1, 2)
#         temp = self.conv1(X)
#         # print('temp', temp.shape)
#         temp2 = torch.sigmoid(self.conv2(X))
#         # temp2 = torch.tanh(self.conv2(X))
#         # print('temp2', temp2.shape)
#         out = F.relu(temp + self.conv3(X))
#         # Convert back from NCHW to NHWC
#         out = out.permute(0, 2, 3, 1)
#         return out


# class STGCNBlock(nn.Module):
#     """
#     Neural network block that applies a temporal convolution on each node in
#     isolation, followed by a graph convolution, followed by another temporal
#     convolution on each node.
#     """

#     def __init__(self, in_channels, spatial_channels, out_channels,
#                  num_nodes):
#         """
#         :param in_channels: Number of input features at each node in each time
#         step.
#         :param spatial_channels: Number of output channels of the graph
#         convolutional, spatial sub-block.
#         :param out_channels: Desired number of output features at each node in
#         each time step.
#         :param num_nodes: Number of nodes in the graph.
#         """
#         super(STGCNBlock, self).__init__()
#         self.temporal1 = TimeBlock(in_channels=in_channels,
#                                    out_channels=out_channels)
#         self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
#                                                      spatial_channels))
#         self.temporal2 = TimeBlock(in_channels=spatial_channels,
#                                    out_channels=out_channels)
#         self.batch_norm = nn.BatchNorm2d(num_nodes)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.Theta1.shape[1])
#         self.Theta1.data.uniform_(-stdv, stdv)

#     def forward(self, X, A_hat):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         :return: Output data of shape (batch_size, num_nodes,
#         num_timesteps_out, num_features=out_channels).
#         """
#         t = self.temporal1(X)
#         lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
#         # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
#         t2 = F.relu(torch.matmul(lfs, self.Theta1))
#         t3 = self.temporal2(t2)
#         return self.batch_norm(t3)
#         # return t3