import torch.profiler
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

    Net = FTAnet()

    Net = torch.nn.DataParallel(Net, device_ids=gid)
    if pretrained is not None:
        Net.load_state_dict(torch.load(pretrained), strict=False)

    if gid is not None:
        Net.to(device=device)
    else:
        Net.cpu()
    Net.float()

    test_list = load_list(path=test, mode='test')

    tick = time.time()

    Net.eval()
    eval_arr = np.zeros(5, dtype=np.double)
    with torch.no_grad():
        for i in range(len(test_list)):
            X_test, y_test = load_data(test_list[i])
            pred_list = []
            if gid is not None:
                splits_x = torch.split(X_test, split_size_or_sections=16)
                for i, split in enumerate(splits_x):
                    pred_list.append(Net(split.cuda())[0])
                pred = torch.cat(pred_list, dim=0)

            else:
                splits_x = torch.split(X_test, split_size_or_sections=16)
                for i, split in enumerate(splits_x):
                    pred_list.append(Net(split)[0])
                pred = torch.cat(pred_list, dim=0)
            est_freq = pred2res(pred).flatten()
            ref_freq = y2res(y_test).flatten()
            time_series = np.arange(len(ref_freq)) * 0.01
            eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)

        eval_arr /= len(test_list)


        print("----------------------")
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                       eval_arr[2], eval_arr[3],
                                                                                       eval_arr[4]))


def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-test', '--test_list_path',
                   help='the path of test data list (default: %(default)s)',
                   type=str, default='./test_05_npy.txt')
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

    train(args.test_list_path, args.gpu_index, pretrained_model)
