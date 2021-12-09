import os
import scipy.io
import numpy as np
from collections import OrderedDict
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))



class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class ArcMarginProduct(nn.Module):
    def __init__(self, s=31.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # cos(theta+m):monotonic decreasing while theta in [0,180]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m)

    def forward(self, cosine, label):
        sine =torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta+m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        onehot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        onehot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (onehot * phi) + ((1.0 - onehot) * cosine)
        output = self.s * output

        return output

# class Mish(nn.Module):
#     def __init__(self):
#         super(Mish,self).__init__()
#
#     def forward(self, x):
#         x = x * (torch.tanh(F.softplus(x)))
#         return x

class MDNet(nn.Module):
    def __init__(self, model_path1=None, K=1, init_weights=True):
        super(MDNet, self).__init__()
        self.K = K
        # ****************RGB_para****************
        self.RGB_para1_3x3 = nn.Sequential(OrderedDict([
            ('Rconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2),

                                     nn.ReLU(), #Mish(),
                                     nn.BatchNorm2d(96),
                                     nn.Dropout(0.5),
                                     # LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2)
                                     ))]))

        self.RGB_para2_1x1 = nn.Sequential(OrderedDict([
            ('Rconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2),

                                     nn.ReLU(),  # Mish(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     # LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))

             )]))

        self.RGB_para3_1x1 = nn.Sequential(OrderedDict([
            ('Rconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2),

                                     nn.ReLU(),  # Mish(),
                                     nn.BatchNorm2d(512),
                                     nn.Dropout(0.5),
                                     # LRN()
                                     )

             )]))

        # *********T_para**********************
        self.T_para1_3x3 = nn.Sequential(OrderedDict([
            ('Tconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2),

                                     nn.ReLU(),  # Mish(),
                                     nn.BatchNorm2d(96),
                                     nn.Dropout(0.5),
                                     # LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))
             )]))

        self.T_para2_1x1 = nn.Sequential(OrderedDict([
            ('Tconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2),

                                     nn.ReLU(),  # Mish(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     # LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))

             )]))

        self.T_para3_1x1 = nn.Sequential(OrderedDict([
            ('Tconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2),

                                     nn.ReLU(),  # Mish(),
                                     nn.BatchNorm2d(512),
                                     nn.Dropout(0.5),
                                     # LRN()
                                     )

             )]))

        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),  # Mish(),
                                    # LRN(),
                                    nn.BatchNorm2d(96),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),  # Mish(),
                                    # LRN(),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(1024 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))

        # self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
        #                                              nn.Linear(512, 2)) for _ in range(K)])



        if init_weights:
            self._initialize_weights()

        self.weight = []
        for i in range(K):
            self.weight.append(nn.Parameter(torch.Tensor(2, 512).cuda()))
                # self.weight.append(nn.Parameter(torch.Tensor(2, 32).cuda()))
            nn.init.xavier_uniform_(self.weight[i])

        self.conv1x1_Tk = nn.Conv2d(512, 32, 1, 1)
        self.conv1x1_Tv = nn.Conv2d(512, 512, 1, 1)
        self.conv1x1_Rv = nn.Conv2d(512, 512, 1, 1)
        self.conv1x1_Rq = nn.Conv2d(512, 32, 1, 1)

        self.conv1x1_Tk = self.conv1x1_Tk.cuda()
        self.conv1x1_Tv = self.conv1x1_Tv.cuda()
        self.conv1x1_Rv = self.conv1x1_Rv.cuda()
        self.conv1x1_Rq = self.conv1x1_Rq.cuda()

        if model_path1 is not None:
            if os.path.splitext(model_path1)[1] == '.pth':
                self.load_model(model_path1)
            elif os.path.splitext(model_path1)[1] == '.mat':
                self.load_mat_model(model_path1)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path1))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()

        # **********************RGB*************************************
        for name, module in self.RGB_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para3_1x1.named_children():
            append_params(self.params, module, name)

        # **********************T*************************************
        for name, module in self.T_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para3_1x1.named_children():
            append_params(self.params, module, name)

        # **********************conv*fc*************************************
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)

        for k, para in enumerate(self.weight):
            #append_params(self.params, module, 'fc6_%d' % (k))
            self.params['weight_' + str(k)] = para

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, xR=None, xT=None, feat=None, k=0, in_layer='conv1', out_layer='fc6'):

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                if name == 'conv1':
                    feat_T = self.T_para1_3x3(xT)
                    feat_R = self.RGB_para1_3x3(xR)
                    feat_MT = module(xT)
                    feat_MR = module(xR)

                    featT = feat_MT + feat_T
                    featR = feat_MR + feat_R

                if name == 'conv2':
                    feat_T = self.T_para2_1x1(featT)
                    feat_R = self.RGB_para2_1x1(featR)
                    feat_MT = module(featT)
                    feat_MR = module(featR)

                    featR = feat_MR + feat_R
                    featT = feat_MT + feat_T

                if name == 'conv3':
                    feat_T = self.T_para3_1x1(featT)
                    feat_R = self.RGB_para3_1x1(featR)
                    feat_MT = module(featT)
                    feat_MR = module(featR)

                    featR = feat_MR + feat_R
                    featT = feat_MT + feat_T
                    #feat = torch.cat((featR, featT), 1)# train 1:1   #test 1:1.4

                    # with 1x1 conv
                    feat_T1 = self.conv1x1_Tk(featT)
                    feat_Tv = self.conv1x1_Tv(featT)
                    feat_Rv = self.conv1x1_Rv(featR)
                    feat_R1 = self.conv1x1_Rq(featR)

                    feat_T2 = feat_T1.view(feat_T1.size()[0], -1, feat_T1.size()[3] * feat_T1.size()[2]).permute(0, 2,
                                                                                                                 1)  # BxNxC
                    feat_R2 = feat_R1.view(feat_R1.size()[0], -1, feat_R1.size()[3] * feat_R1.size()[2])  # BxCxN

                    energy = torch.bmm(feat_T2, feat_R2)  # Bx(N)x(N)
                    softmax = nn.Softmax(dim=-1)
                    attention = softmax(energy)

                    feat_T3 = feat_Tv.view(feat_Tv.size()[0], -1, feat_Tv.size()[3] * feat_Tv.size()[2])
                    feat_R3 = feat_Rv.view(feat_Rv.size()[0], -1, feat_Rv.size()[3] * feat_Rv.size()[2])
                    feat_T_out = torch.bmm(feat_T3, attention.permute(0, 2, 1))
                    feat_T_out = feat_T_out.view(featT.size())

                    feat_T_out = featT + feat_T_out

                    feat_R_out = torch.bmm(feat_R3, attention.permute(0, 2, 1))
                    feat_R_out = feat_R_out.view(featR.size())

                    #without 1x1 conv
                    # feat_T2 = feat_T.view(feat_T.size()[0], -1, feat_T.size()[3] * feat_T.size()[2]).permute(0, 2,
                    #                                                                                  1)  # BxNxC
                    # feat_R2 = feat_R.view(feat_R.size()[0], -1, feat_R.size()[3] * feat_R.size()[2])  # BxCxN
                    #
                    # energy = torch.bmm(feat_T2, feat_R2)  # Bx(N)x(N)
                    # softmax = nn.Softmax(dim=-1)
                    # attention = softmax(energy)
                    #
                    # feat_T3 = feat_T.view(feat_T.size()[0], -1, feat_T.size()[3] * feat_T.size()[2])
                    # feat_R3 = feat_R.view(feat_R.size()[0], -1, feat_R.size()[3] * feat_R.size()[2])
                    # feat_T_out = torch.bmm(feat_T3, attention.permute(0, 2, 1))
                    # feat_T_out = feat_T_out.view(featT.size())
                    #
                    # feat_T_out = featT + feat_T_out
                    #
                    # feat_R_out = torch.bmm(feat_R3, attention.permute(0, 2, 1))
                    # feat_R_out = feat_R_out.view(featR.size())
                    #
                    #
                    feat_R_out = featR + feat_R_out

                    feat = torch.cat((feat_T_out, feat_R_out), 1)
                    feat = feat.contiguous().view(feat.size(0), -1)
                if name == 'fc4':
                    feat = module(feat)

                if name == 'fc5':
                    feat = module(feat)

                if name == out_layer:
                    return feat

        #feat = self.branches[k](feat)  # (batch_size, hidden_size)
        feat = feat.view(feat.size(0), -1)
        # print(self.weight[k])             # (num_classes, hidden_size)
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight[k]))
        if out_layer == 'fc6':
            return cosine
        elif out_layer == 'fc6_softmax':
            return F.softmax(cosine, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers, strict=False)

        para1_layers = states['RGB_para1_3x3']
        self.RGB_para1_3x3.load_state_dict(para1_layers, strict=True)
        para2_layers = states['RGB_para2_1x1']
        self.RGB_para2_1x1.load_state_dict(para2_layers, strict=True)
        para3_layers = states['RGB_para3_1x1']
        self.RGB_para3_1x1.load_state_dict(para3_layers, strict=True)

        para1_layers = states['T_para1_3x3']
        self.T_para1_3x3.load_state_dict(para1_layers, strict=True)
        para2_layers = states['T_para2_1x1']
        self.T_para2_1x1.load_state_dict(para2_layers, strict=True)
        para3_layers = states['T_para3_1x1']
        self.T_para3_1x1.load_state_dict(para3_layers, strict=True)

        print('load finish pth!!!')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

        print('load mat finish!')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# class BinaryLoss(nn.Module):
#     def __init__(self):
#         super(BinaryLoss, self).__init__()
#
#     def forward(self, pos_score, neg_score):
#         pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
#         neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
#
#         loss = pos_loss.sum() + neg_loss.sum()
#         return loss

class Arcfaceloss(nn.Module):
    def __init__(self):
        super(Arcfaceloss, self).__init__()
        self.ArcMargin = ArcMarginProduct()
        self.classifier = nn.CrossEntropyLoss()
    def forward(self, pos_score, neg_score):
        score = torch.cat((pos_score,neg_score),0)
        gt_pos = torch.ones(pos_score.shape[0], 1).cuda()
        gt_neg = torch.zeros(neg_score.shape[0], 1).cuda()
        gt = torch.cat((gt_pos,gt_neg),0).squeeze()
        output = self.ArcMargin(score, gt)
        loss = self.classifier(output,gt.long())

        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        # return prec.data[0]

        return prec.data
