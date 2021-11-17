import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class mse_loss(nn.Module):
    def __init__(self, w1, w2):
        super(mse_loss, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, input1, input2, target):
        loss1 = self.w1 * F.mse_loss(input1, target)
        loss2 = self.w2 * F.mse_loss(input1, input2)
        loss = loss1 + loss2
        print('loss', loss1, loss2)
        return loss