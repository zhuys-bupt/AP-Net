from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='AP-Net')
parser.add_argument('--model', default='gwcnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')

parser.add_argument('--device', default='cuda', help='device to use for training / testing')

# parse arguments
args = parser.parse_args()
# get device
device = torch.device(args.device)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=2, drop_last=False)

# model
model = __models__[args.model](args.maxdisp)

if device.type == 'cuda':
    model = nn.DataParallel(model)
    model.cuda()

param_sum = 0
for param in model.parameters():
    param_sum += param.nelement()
print("Parameters: %.2f M" % (param_sum / 1024 / 1024))


def test(epoch_idx):
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        scalar_outputs = test_sample(sample)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs

    avg_test_scalars = avg_test_scalars.mean()
    print("Epoch {", epoch_idx, "},  avg_test_scalars", avg_test_scalars)

    gc.collect()
    # empty cache
    torch.cuda.empty_cache()


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    if device.type == 'cuda':
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(scalar_outputs)


if __name__ == '__main__':
    if args.loadckpt.endswith(".ckpt"):
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'])
        epoch_idx = int(args.loadckpt.split('_')[-1].split('.')[0])
        # test
        test(epoch_idx)
    else:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = [fn for fn in os.listdir(args.loadckpt) if fn.endswith(".ckpt")]
        all_saved_ckpts = sorted(all_saved_ckpts)
        for ckpt in all_saved_ckpts:
            # use the latest checkpoint file
            loadckpt = os.path.join(args.loadckpt, ckpt)
            # load the checkpoint file specified by args.loadckpt
            print("loading model {}".format(loadckpt))
            state_dict = torch.load(loadckpt)
            model.load_state_dict(state_dict['model'])
            epoch_idx = int(ckpt.split('_')[-1].split('.')[0])
            # test
            test(epoch_idx)
