#!/usr/bin/env python
#coding=utf-8

from modules import dataLoader as dld
from torch.utils.data import DataLoader
import logging

logging.getLogger().setLevel(logging.INFO)



def init_dataSet(config,device):
    insize = config.getint('dataset','insize')

    # init training set
    trainlist = config['train']['list']
    trainset  = dld.ImageListDatasetPreload(trainlist,insize,device)
    numTrain  = len(trainset)

    logging.info('-'*25)
    logging.info('training set contains %d samples'%numTrain)

    nworker = config.getint('dataset','nworker')
    tbatch = config.getint('train','bsize')
    #trainloader = DataLoader(trainset,batch_size=tbatch,\
    #        shuffle=True,num_workers=nworker,pin_memory=True)
    trainloader = DataLoader(trainset,batch_size=tbatch,\
            shuffle=True,num_workers=nworker)

    logging.info('training loader is already')
    logging.info('-'*25)

    ''''
    # init validing set
    testlist = config['val']['list']
    vbatch = config.getint('val','bsize')
    testset  = dld.ImageListDataset(testlist,insize,\
            True,True)
    numTest  = len(testset)

    logging.info('testing set contains %d samples'%numTest)

    testloader = DataLoader(testset,batch_size=vbatch,\
            shuffle=False,num_workers=nworker)

    logging.info('testing loader is already')
    '''

    dataloader = {'train':trainloader}
    datasize = {'train':numTrain}
    #dataloader = {'train':trainloader,'val':testloader}
    #datasize = {'train':numTrain,'val':len(testset)}

    return dataloader,datasize
