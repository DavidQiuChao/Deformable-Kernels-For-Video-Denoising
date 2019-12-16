#!/usr/bin/env python
#coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_node_value(ims,bid,cid,hid,wid):
    b,c,h,w = ims.size()
    #get mask of pixels outside the matrix(frames) 
    #caused by adding offset
    omask = ((0 <= cid).int() + (cid < c).int()\
            + (0 <= hid).int() + (hid < h).int()\
            + (0 <= wid).int() + (wid < w).int())\
            != 6
    #set outside index to 0 for avoiding index out of bounds
    cid[omask]=0
    hid[omask]=0
    wid[omask]=0
    #select nodes 
    nodes = ims[bid,cid,hid,wid]
    #ignoring outside pixels by seting value to 0 
    nodes[omask] = 0
    return nodes


def sample_pixels(ims,offsets):
    '''
    sample pixels from frames
    new pixels locations which added offsets
    are selected by trilinear interpolation
    '''
    b,c,h,w = ims.size()
    # numble of sample pixels
    N = offsets.size(1)//3
    device = ims.device.type

    #initial sample pixels result
    samples = torch.zeros(b,N,h,w,device=device)

    #make grids index matrix
    h_loc,w_loc = torch.meshgrid(torch.arange(start=0,end=h),\
            torch.arange(start=0,end=w))

    #reshape grids to fit frames shape for easy indexing 
    h_loc = h_loc.view(1,1,h,w).expand(b,N,h,w).long().to(device)
    w_loc = w_loc.view(1,1,h,w).expand(b,N,h,w).long().to(device)
    t_loc = torch.Tensor([c//2]).view(1,1,1,1).expand(b,N,h,w).\
            long().to(device)
    bid = torch.arange(b).view(b,1,1,1).expand(b,N,h,w).long()

    # find the corners of a cubic for trilinear interpolation
    floor,ceil = torch.floor, torch.ceil
    corners = (
        (floor, floor, floor),
        (floor, floor, ceil),
        (floor, ceil, floor),
        (floor, ceil, ceil),
        (ceil, floor, floor),
        (ceil, floor, ceil),
        (ceil, ceil, floor),
        (ceil, ceil, ceil),
    )
    #trilinear interpolation for 8 neighbor nodes 
    #import pdb;pdb.set_trace()
    for ct,ch,cw in corners:
        t_off = ct(offsets[:,0::3,...])
        h_off = ch(offsets[:,1::3,...])
        w_off = cw(offsets[:,2::3,...])
        samples += get_node_value(\
                ims,\
                bid,\
                t_off.long()+t_loc,\
                h_off.long()+h_loc,\
                w_off.long()+w_loc)\
                * (1 - torch.abs(t_off - offsets[:, 0::3, ...]))\
                * (1 - torch.abs(h_off - offsets[:, 1::3, ...]))\
                * (1 - torch.abs(w_off - offsets[:, 2::3, ...]))

    return samples


def convolution_3D(samples,kernels):
    #convolved the samples with kernels at corresponding positions
    b,N,h,w = samples.size()
    kernels = kernels.view(b,N,h,w)
    return torch.sum(samples*kernels,dim=1,keepdim=False)




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frames = torch.arange(81920).reshape(1,5,128,128).float().to(device)
    frames = torch.randn(1,5,128,128).to(device)
    offset = torch.ones(1,81,128,128).to(device)
    samples = sample_pixels(frames,offset)
    print(samples)
    #unfold = convolution_3D(samples, torch.ones(2,1,3,3))
    #print(unfold)



