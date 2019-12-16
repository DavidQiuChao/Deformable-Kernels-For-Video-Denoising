#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F



class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:#a approximation way for quickly
            return torch.abs((l - r)[..., 0:w, 0:h])\
                    + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(torch.pow((l - r)[..., 0:w, 0:h], 2)\
                    + torch.pow((u - d)[..., 0:w, 0:h], 2))


class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss,self).__init__()
        self.l1Loss = nn.L1Loss()
        self.l2Loss = nn.MSELoss()
        self.gradient = TensorGradient()

    def forward(self,pred,gt):
        return self.l1Loss(self.gradient(pred),\
                    self.gradient(gt))\
                    + self.l2Loss(pred,gt)


class AnnealLoss(nn.Module):
    def __init__(self,ita=100,gama=0.9998):
        super(AnnealLoss,self).__init__()
        self.loss = BasicLoss()
        self.ita = ita
        self.gama = gama

    def forward(self,step,preds,gt):
        loss = 0
        for i in range(preds.size(1)):
            loss += self.loss(
                    torch.unsqueeze(preds[:,i,...],1),
                    gt)
        loss /= preds.size(1)

        return self.ita*self.gama**step*loss


def gammaCorrection(x):
    alpha = 0.055
    gamma = 1/2.4
    mask = (x<=0.0031308)
    x[mask] *= 12.92
    x[~mask] = (1+alpha)*torch.pow(x[~mask]+0.001,gamma)-alpha
    return x


class totalLoss(nn.Module):
    def __init__(self):
        super(totalLoss,self).__init__()
        self.base = BasicLoss()
        self.anneal = AnnealLoss()

    def forward(self,pred,preds,gt,step):
        pred = gammaCorrection(pred)
        gt = gammaCorrection(gt)
        preds = gammaCorrection(preds)
        return self.base(pred,gt),self.anneal(step,preds,gt)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = torch.randn(1,1,128,128).float().to(device)
    preds = torch.randn(1,4,128,128).float().to(device)
    gt = torch.randn(1,1,128,128).float().to(device)
    tloss = totalLoss()
    bloss,aloss = tloss(pred,preds,gt,1)
    print(loss)


