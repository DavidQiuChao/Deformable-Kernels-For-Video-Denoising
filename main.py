#!/usr/bin/env python
#coding=utf-8

import os
import time
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import train as tp
from modules import prepData as prp
from modules.netArcs import DeformabelKPN
import logging

logging.getLogger().setLevel(logging.INFO)


def load_pretrained_model(pmodel,model):
    logging.info ("load pretrained-model")
    checkpoint = torch.load(pmodel)
    checkpoint = {k.replace('module.', ''):v for k,v in checkpoint.items()}
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(checkpoint)
    return model

def main(args):
    #read params config
    cf = args.config
    config = configparser.ConfigParser()
    config.read(cf)
    gpu = config['dataset']['gpu']
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #init net
    net = DeformabelKPN(5,27,3)
    if config.has_option('net','weights'):
        pmodel = config['net']['weights']
        net = load_pretrained_model(pmodel,net)
    net.to(device)

    #init dataset
    data_loader,datasize = prp.init_dataSet(config,device)
    logging.info('dataset loading finish')

    #set optimizer function
    lr = config.getfloat('net','lr')
    gamma = config.getfloat('net','gamma')
    stepsize = int(config['net']['stepsize'])
    #optimizer = optim.Adam(net.parameters(),\
    #        lr=lr,betas=(0.9,0.99))
    optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':lr}],\
            lr=lr,betas=(0.9,0.99))
    if config.has_option('net','optimizer'):
        popt = config['net']['optimizer']
        optimizer.load_state_dict(torch.load(popt))

    #set learning rate method
    scheduler = optim.lr_scheduler.StepLR(optimizer,stepsize,gamma)

    #trainning
    logging.info('start training')
    pic = config['save']['cmppic']
    mepoch = config.getint('net','maxepoch')
    its = config.getint('net','itersize')
    sinter = config.getint('save','step')
    outdir = config['save']['model']
    tp.train_model(outdir,net,optimizer,scheduler,\
        data_loader,datasize,device,mepoch,its,\
        sinter,pic,config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',help='params config file')
    args = parser.parse_args()
    main(args)
