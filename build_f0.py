import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import os

from utils import *
from ftanet import FTAnet
import time

import argparse

class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
    

def train(test, gid, pretrained=None):

    torch.backends.cudnn.enabled = False
    gid = list(map(int, gid.split(",")))
    device = torch.device("cuda:{}".format(gid[0]))
    # gid=None
    Net = FTAnet()
    Net = torch.nn.DataParallel(Net, device_ids=gid)
    if pretrained is not None:
        Net.load_state_dict(torch.load(pretrained))

    if gid is not None:
        Net.to(device=device)
    else:
        Net.cpu()
    Net.float()


    test_list = load_list(path=test, mode='test')

    Net.eval()

    with torch.no_grad():
        for i in range(len(test_list)):
            print(test_list[i])
            X_test = load_data(test_list[i])

            if gid is not None:
                pred = Net(X_test.cuda())
            else:
                pred = Net(X_test)
            est_freq = pred2res(pred[0][:,0, :,:].cpu()).flatten()
            print('NO.{},length of est_freq {}'.format(test_list[i],est_freq.shape))

            time_series = np.arange(len(est_freq)) * 0.01
            print('NO.{},length of time_series {}'.format(test_list[i], time_series.shape))

            time_size = len(est_freq)
            print('NO.{},length of time_size {}'.format(test_list[i], time_size))
            pr_freq = est_freq.reshape(time_size, 1)
            pr_time = time_series.reshape(time_size, 1)

            f0_pr = np.concatenate((pr_time, pr_freq), axis=1)

            np.savetxt(os.path.join('./data', 'myf02', test_list[i]).replace('.npy', '.txt'), f0_pr, delimiter='\t', fmt='%.3f')



def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-test', '--test_list_path',
                   help='the path of test data list (default: %(default)s)',
                   type=str, default='./test_10_npy.txt')
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                   type=str, default="0")
    p.add_argument('-pm', '--pretrained_model',
                   help='the path of pretrained model (Transformer or Streamline) (default: %(default)s)',
                   type=str)

    return p.parse_args()


if __name__ == '__main__':
    args = parser()
    # args.gpu_index = None
    gid = args.gpu_index
    gid = list(map(int, gid.split(",")))[0]

    pretrained_model = None
    print(pretrained_model)

    train(args.test_list_path, args.gpu_index, pretrained_model)

