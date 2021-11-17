import time
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from model.stgat2 import STGAT
from loss.MSELoss import mse_loss
from loss.MAPELoss import MAPELoss

parser = argparse.ArgumentParser()
# parser.add_argument('--graph_signal_matrix_filename', type=str, default='data/METR-LA/data2.npz')
parser.add_argument('--data', type=str, default='data/METR-LA/')
parser.add_argument('--adj_filename', type=str, default='data/METR-LA/adj_mx_dijsk.pkl')
parser.add_argument('--params_dir', type=str, default='experiment_METR_LA')
parser.add_argument('--num_of_vertices', type=int, default=207)
parser.add_argument('--num_of_features', type=int, default=2)
parser.add_argument('--points_per_hour', type=int, default=12)
parser.add_argument('--num_for_predict', type=int, default=12)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--print_every', type=float, default=100)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--adjtype', type=str, default='symnadj')
parser.add_argument('--early_stop_maxtry', type=int, default=1000)
parser.add_argument('--cuda', action='store_true', help='use CUDA training.')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
print(f'Training configs: {args}')


def weight_schedule(epoch, max_val=10, mult=-5, max_epochs=100):
    if epoch == 0:
        return 0.
    w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
    w = float(w)
    if epoch > max_epochs:
        return max_val
    return w

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adj_filename, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    for iter, (trainx, trainy, trainx_) in enumerate(dataloader['train_loader']):
        x = torch.where(trainx>1, torch.zeros_like(trainx), trainx)
        print(x)

if __name__ == "__main__":
    a = torch.randn(3, 3)
    print(a)
    std = torch.std(a)
    a = a / std
    b = torch.mul(a, a)
    print(b)
    c = torch.exp(b)