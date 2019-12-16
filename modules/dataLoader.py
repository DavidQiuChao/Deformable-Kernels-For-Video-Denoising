#!/usr/bin/env python
#encoding=utf-8

import os
import logging
import time
import torch
import numpy as np
from tqdm import tqdm
from skimage import io
from multiprocessing import Pool
from torch.utils.data import Dataset,DataLoader
from concurrent.futures import ThreadPoolExecutor

logging.getLogger().setLevel(logging.INFO)



def load_list(listPath):
    with open(listPath,'r') as f:
        lines = [i.strip() for i in f.readlines()]
    return lines


def random_crop(ims,gt,ps=512):
    H = ims.shape[0]
    W = ims.shape[1]
    xs = np.random.randint(0,W-ps)
    ys = np.random.randint(0,H-ps)
    inPatch = ims[ys:ys+ps,xs:xs+ps,:]
    gtPatch = gt[ys:ys+ps,xs:xs+ps,:]
    del ims,gt
    return inPatch,gtPatch


def random_flip(inPatch,gtPatch):
    if np.random.randint(2,size=1)[0] == 1:  # random flip 
        inPatch = torch.flip(inPatch,[0])
        gtPatch = torch.flip(gtPatch,[0])
    if np.random.randint(2,size=1)[0] == 1:
        inPatch = torch.flip(inPatch,[1])
        gtPatch = torch.flip(gtPatch,[1])
    if np.random.randint(2,size=1)[0] == 1:  # random transpose 
        inPatch = inPatch.permute(1,0,2)
        gtPatch = gtPatch.permute(1,0,2)
    return inPatch,gtPatch


def load_each(path):
    gtpath = path.strip()+'_lb.npy'
    gt = np.load(gtpath)
    return gt


def preload_data(paths,nps=8):
    num = len(paths)
    pool = Pool(processes=nps)
    out = []
    logging.info('start loading data...')
    for path in tqdm(paths):
        out.append([pool.apply_async(load_each,(path,)),path])
    pool.close()
    pool.join()
    logging.info('data loading done...')
    return out


def read_all(path,device):
    fpaths = [path+'_%d.npy'%i for i in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        outs = pool.map(np.load,fpaths)
        for i,im in enumerate(outs):
            im = torch.from_numpy(im).to(device)
            if i == 0:
                ims = im
                continue
            ims = torch.cat((ims,im),2)
    return torch.squeeze(ims)


def read_src(path):
    fpaths = [path+'_%d.npy'%i for i in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        outs = pool.map(np.load,fpaths)
        for i,im in enumerate(outs):
            if i == 0:
                ims = im
                continue
            ims = np.concatenate((ims,im),2)
    return np.squeeze(ims)


def preload_all_data(paths,nps=16):
    num = len(paths)
    pool = Pool(processes=nps)
    out = []
    logging.info('start loading data...')
    for path in tqdm(paths):
        src = read_src(path)
        out.append([src,pool.apply_async(load_each,(path,)),path])
    pool.close()
    pool.join()
    logging.info('data loading done...')
    return out


class ImageListDatasetGPU(Dataset):
    def __init__(self,imlist,csize,device,transform=True,val=False):
        self.impaths = load_list(imlist)
        self.transform = transform
        self.csize = csize
        self.val = val
        self.infs = preload_data(self.impaths,16)
        self.device = device

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self,idx):
        t0 = time.time()
        gtob,path = self.infs[idx]
        gt = gtob.get()
        gt = torch.from_numpy(gt).to(self.device)
        gt = torch.squeeze(gt,3)
        t1 = time.time()
        ims = read_all(path,self.device)
        t2 = time.time()
        gtt = (t1-t0)%60
        srct = (t2-t1)%60
        logging.info('gt:{:.6f}s,src:{:.6f}s'.format(gtt,srct))
        if self.transform:
            ims,gt = random_crop(ims,gt,self.csize)
            if not self.val:
                ims,gt = random_flip(ims,gt)
        return ims,gt


class ImageListDatasetPreload(Dataset):
    def __init__(self,imlist,csize,device,transform=True,val=False):
        self.impaths = load_list(imlist)
        self.transform = transform
        self.csize = csize
        self.val = val
        self.infs = preload_all_data(self.impaths,16)
        self.device = device

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self,idx):
        ims,gtob,path = self.infs[idx]
        gt = gtob.get()
        gt = torch.from_numpy(gt).to(self.device)
        gt = torch.squeeze(gt,3)
        ims = torch.from_numpy(ims).to(self.device)
        if self.transform:
            ims,gt = random_crop(ims,gt,self.csize)
            if not self.val:
                ims,gt = random_flip(ims,gt)
        return ims,gt


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import time
    tlist = '../tmp.list'
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = ImageListDatasetGPU(tlist,512,device)
    dtLoader = DataLoader(data,batch_size=1,\
            shuffle=True,num_workers=0)
    t0 = time.time()
    for ims,gth in data:
        t1 = time.time()
        print((t1-t0)%60)
        t0 = t1
    print('all done')
