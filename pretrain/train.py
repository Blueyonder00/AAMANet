import os
import sys
import pickle
import time
import random

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from modules.MANet3x1x1_IC_A_arcface import *
from options import *
from tensorboardX import SummaryWriter
from PIL import Image

# ********************************************set dataset path ********************************************
# ********************************************set seq list .pkl file path  ********************************
img_home = "../dataset/airplane-train-ori"
data_path1 = '../data/train-1.pkl'
data_path2 = '../data/train-2.pkl'


# *********************************************************************************************************
def init_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enable = True


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    ## Init dataset ##
    with open(data_path1, 'rb') as fp1:
        data1 = pickle.load(fp1)
    with open(data_path2, 'rb') as fp2:
        data2 = pickle.load(fp2)
    K1 = len(data1)
    dataset1 = [None] * K1

    for k, ((seqname1, seq1), (seqname2, seq2)) in enumerate(zip(sorted(data1.items()), sorted(data2.items()))):
        img_dir1 = []
        img_dir2 = []
        img_list1 = seq1['images']
        gt1 = seq1['gt']
        img_list2 = seq2['images']
        for i in [15,16,17]:
            img_dir = os.path.join(img_home, seqname1 + '/channel' + str(i).zfill(2))
            img_dir1.append(img_dir)
        for j in [11,20,25]:
            img_dir = os.path.join(img_home, seqname2 + '/channel' + str(j).zfill(2))
            img_dir2.append(img_dir)

        dataset1[k] = RegionDataset(img_dir1, img_list1, img_dir2, img_list2, gt1, opts)

    ## Init model ##
    model = MDNet(model_path1=None, K=K1)  # opts['init_model_path']
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    ## Init criterion and optimizer ##
    criterion = Arcfaceloss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    if not os.path.exists(opts['rgb_log_dir']):
        os.makedirs(opts['rgb_log_dir'])
    summary_writer = SummaryWriter(opts['rgb_log_dir'])

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print("==== Start Cycle %d ====" % i)
        k_list = np.random.permutation(K1)
        prec = np.zeros(K1)
        for j, k in enumerate(k_list):
            # if k!=55: continue
            tic = time.time()
            pos_regions1, neg_regions1, idx1, pos_regions2, neg_regions2, idx2 = \
                dataset1[k].next()
            # print(dataset1[k].pointer)
            pos_regions1 = Variable(pos_regions1)
            neg_regions1 = Variable(neg_regions1)
            pos_regions2 = Variable(pos_regions2)
            neg_regions2 = Variable(neg_regions2)

            if opts['use_gpu']:
                pos_regions1 = pos_regions1.cuda()
                neg_regions1 = neg_regions1.cuda()
                pos_regions2 = pos_regions2.cuda()
                neg_regions2 = neg_regions2.cuda()

            pos_score = model(pos_regions1, pos_regions2, k=k)
            neg_score = model(neg_regions1, neg_regions2, k=k)

            loss = criterion(pos_score, neg_score)
            step = i * len(k_list) + j
            summary_writer.add_scalar('train/cls_loss', loss.data, step)

            model.zero_grad()

            loss.backward()
            # print(model.weight[k].data)
            # torch.nn.utils.clip_grad_value_(model.weight[k],opts['clip_value'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])

            optimizer.step()
            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print("Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % (i, j, k, loss.item(), prec[k], toc))
        cur_prec = prec.mean()
        print("Mean Precision: %.3f" % cur_prec)
        if i>70 and cur_prec>best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                'shared_layers': model.layers.state_dict(),

                'RGB_para1_3x3': model.RGB_para1_3x3.state_dict(),
                'RGB_para2_1x1': model.RGB_para2_1x1.state_dict(),
                'RGB_para3_1x1': model.RGB_para3_1x1.state_dict(),

                'T_para1_3x3': model.T_para1_3x3.state_dict(),
                'T_para2_1x1': model.T_para2_1x1.state_dict(),
                'T_para3_1x1': model.T_para3_1x1.state_dict(),

                'conv1x1_Rq': model.conv1x1_Rq.state_dict(),
                'conv1x1_Rv': model.conv1x1_Rv.state_dict(),
                'conv1x1_Tk': model.conv1x1_Tk.state_dict(),
                'conv1X1_Tv': model.conv1x1_Tv.state_dict(),
            }

            print("Save model to %s.pth" % opts['model_path'] + str(i) + '.pth')
            torch.save(states, opts['model_path'] + str(i) + '.pth')
        if opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    init_seed(125)
    train_mdnet()
