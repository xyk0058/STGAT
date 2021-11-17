import time
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from model.stgat2_testing import STGAT, STGATModel
from loss.MSELoss import mse_loss
from loss.MAPELoss import MAPELoss

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/METR-LA/')
# parser.add_argument('--data', type=str, default='data/PEMS-BAY/')
parser.add_argument('--adj_filename', type=str, default='data/METR-LA/adj_mx_dijsk.pkl')
# parser.add_argument('--adj_filename', type=str, default='data/PEMS-BAY/adj_mx_bay.pkl')
# parser.add_argument('--params_dir', type=str, default='experiment_METR_LA')
parser.add_argument('--num_of_vertices', type=int, default=207)#207
# parser.add_argument('--num_of_vertices', type=int, default=325)#207
parser.add_argument('--num_of_features', type=int, default=2)
parser.add_argument('--points_per_hour', type=int, default=12)
parser.add_argument('--num_for_predict', type=int, default=12)
parser.add_argument('--num_of_weeks', type=int, default=1)
parser.add_argument('--num_of_days', type=int, default=1)
parser.add_argument('--num_of_hours', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--print_every', type=float, default=200)
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
    # dataloader2 = util.load_dataset('data/METR-LA/', args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    adj_mx = torch.from_numpy(np.array(adj_mx))[0]
    if args.cuda:
        adj_mx = adj_mx.cuda()
    print('adj', adj_mx.shape)

    with torch.no_grad():
        testnet1 = STGATModel(args.cuda, args.num_of_vertices, args.num_of_features, args.points_per_hour*args.num_of_hours, args.num_for_predict, need_adj=False)
        testnet1.A_hat = nn.Parameter(adj_mx)
        testnet2 = STGATModel(args.cuda, args.num_of_vertices, args.num_of_features, args.points_per_hour*args.num_of_hours, args.num_for_predict, need_adj=True)
        if args.cuda:
            testnet1 = torch.load('model-la/model_net1.pkl').eval()
            testnet2 = torch.load('model-la/model_net2.pkl').eval()
        else:
            testnet1 = torch.load('model-la/model_net1.pkl').cpu().eval()
            testnet2 = torch.load('model-la/model_net2.pkl').cpu().eval()
        
        print('adj', testnet1.A_hat)

        # Testing
        outputs = []
        realy = []
        for iter, (testx, testy) in enumerate(dataloader['test_loader']):
            if args.cuda:
                testx = testx.cuda()
                testy = testy.cuda()
            
            output1 = testnet1.eval().forward(testx, adj_mx)
            output2 = testnet2.eval().forward(testx, adj_mx)
            output = (output1 + output2) / 2

            output = output.permute(0, 3, 1, 2)
            output = output.squeeze()
            outputs.append(output)
            realy.append(testy[:,0,:,:].squeeze())

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy, dim=0)
        if args.cuda:
            yhat = yhat.cuda()
            realy = realy.cuda()

        print("Training finished")

        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = scaler.inverse_transform(realy[:,:,i])

            print('pred', type(pred), pred.shape)

            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

            
            # data = pred

            pred = np.array(pred.cpu().numpy())
            real = np.array(real.cpu().numpy())

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        print('==================================================================================')
        print('\r\n\r\n\r\n')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
