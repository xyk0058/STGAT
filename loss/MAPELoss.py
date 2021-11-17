import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPELoss(nn.Module):

    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, pred, label):
        return torch.mean(torch.abs(pred - label)/torch.abs(label)+1) * 100

