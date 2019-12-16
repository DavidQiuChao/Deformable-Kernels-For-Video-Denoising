#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.offsetNet import OffsetNet
from modules.Conv3DSample import sample_pixels as samp
from modules.Conv3DSample import convolution_3D as conv3D


class WeightsNet(nn.Module):
    def __init__(self,In,Out):
        super(WeightsNet,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(In,64,3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,64,3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,Out,3,1,1),
                nn.ReLU(inplace=True),
                )

    def forward(self,x):
        return self.conv(x)



class DeformabelKPN(nn.Module):
    def __init__(self,nframes=5,nsample=27,ngroup=4):
        super(DeformabelKPN,self).__init__()
        self.nf = nframes
        self.N  = nsample
        self.gp = ngroup
        nweight = nframes + 128 + nsample
        self.offsetNet = OffsetNet(nframes,nsample*3)
        self.weightNet = WeightsNet(nweight,nsample)

    def forward(self,frames):
        b,c,h,w = frames.size()
        feat,offset = self.offsetNet(frames)
        samples = samp(frames,offset)
        feat = torch.cat([frames,feat,samples],dim=1)
        weights = self.weightNet(feat)

        # 3D conv group by group
        npg = self.N//self.gp
        samples_group = torch.split(samples,split_size_or_sections=npg,dim=1)
        weights_group = torch.split(weights,split_size_or_sections=npg,dim=1)

        gp_pred = torch.zeros(b,self.gp,h,w,device=frames.device.type)
        for i in range(self.gp):
            gp_pred[:,i,...] = self.gp * conv3D(samples_group[i],weights_group[i])

        return gp_pred, torch.sum(gp_pred,dim=1,keepdim=True)/self.N



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,5,128,128).to(device)
    net = DeformabelKPN()
    net.to(device)
    net.train()
    gppred,pred = net(x)
    print(gppred.shape,pred.shape)

