#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F



class Basic_Block(nn.Module):
    def __init__(self,In,Out,ks,std,pad):
        super(Basic_Block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(In,Out,ks,std,pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.conv(x)


class Up_Skip_Block(nn.Module):
    def __init__(self,In,Out):
        super(Up_Skip_Block,self).__init__()
        self.up = nn.Upsample(scale_factor=2,\
                mode='bilinear',align_corners=True)
        self.bb = Basic_Block(In,Out,3,1,1)

    def forward(self,x,skip):
        out = self.up(x)
        out = torch.cat([out,skip],1)
        out = self.bb(out)
        return out


class Up_Skip_Block2(nn.Module):
    def __init__(self,In,Out):
        super(Up_Skip_Block2,self).__init__()
        self.up = nn.Upsample(scale_factor=2,\
                mode='bilinear',align_corners=True)
        self.conv = nn.Sequential(
                nn.Conv2d(In,Out,3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(Out,Out,3,1,1),
                nn.ReLU(inplace=True)
            )

    def forward(self,x,skip):
        out = self.up(x)
        out = torch.cat([out,skip],1)
        out = self.conv(out)
        return out


class OffsetNet(nn.Module):
    def __init__(self,In,Out):
        super(OffsetNet,self).__init__()
        self.bb1 = Basic_Block(In,64,3,1,1)
        self.bb2 = Basic_Block(64,128,3,1,1)
        self.bb3 = Basic_Block(128,256,3,1,1)
        self.bb4 = Basic_Block(256,512,3,1,1)
        self.bb5 = Basic_Block(512,512,3,1,1)
        self.up1 = Up_Skip_Block(1024,512)
        self.up2 = Up_Skip_Block(768,256)
        self.up3 = Up_Skip_Block(384,128)
        self.up4 = Up_Skip_Block2(192,128)
        self.cnf = nn.Conv2d(128,Out,3,1,1)

    def forward(self,x):
        feat1  = self.bb1(x)
        feat2  = self.bb2(F.avg_pool2d(feat1,kernel_size=2,stride=2))
        feat3  = self.bb3(F.avg_pool2d(feat2,kernel_size=2,stride=2))
        feat4  = self.bb4(F.avg_pool2d(feat3,kernel_size=2,stride=2))
        feat5  = self.bb5(F.avg_pool2d(feat4,kernel_size=2,stride=2))
        feat6  = self.up1(feat5,feat4)
        feat7  = self.up2(feat6,feat3)
        feat8  = self.up3(feat7,feat2)
        ofeat  = self.up4(feat8,feat1)
        offset = torch.tanh(self.cnf(ofeat))
        return ofeat,offset




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,1,128,128).to(device)
    net = OffsetNet(1,27)
    net.to(device)
    net.train()
    ofeat,offset = net(x)
    print(ofeat.shape,offset.shape)

