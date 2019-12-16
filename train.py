#!/usr/bin/env python
#coding=utf-8

import os
import copy
import time
import logging
import torch
import numpy as np
from PIL import Image
#import scipy.misc
from modules import lossFunc
from utils.train_utils import calculate_psnr, calculate_ssim
from tensorboardX import SummaryWriter

logging.getLogger().setLevel(logging.INFO)


def gammaCorrection(x):
    alpha = 0.055
    gamma = 1/2.4
    mask = (x<=0.0031308)
    x[mask] *= 12.92
    x[~mask] = (1+alpha)*torch.pow(x[~mask]+0.001,gamma)-alpha
    return x


def save_compare_res(pred,gt,ori,root):
    pred = torch.clamp(pred[0,...], 0.0, 1.0)
    ori  = torch.clamp(ori[0,...], 0.0, 1.0)
    out = torch.cat((ori[0,...],pred[0,...]),1)
    out = torch.cat((gt[0,0,...],out),1).cpu().data.numpy()
    image = Image.fromarray(np.uint8(out*255))
    image.save(root+'/cmp.png')


def save_checkpoint(model,optimizer,outdir,epoch):
    sname = os.path.join(outdir,'epoch_%d.pth'%epoch)
    torch.save(model.state_dict(),sname)
    sname = os.path.join(outdir,'opt_%d.pth'%epoch)
    torch.save(optimizer.state_dict(),sname)


def write_log(bloss,aloss,loss,psnr,ssim,iters,log_writer):
    log_writer.add_scalar(
            'basicLoss',bloss,iters)
    log_writer.add_scalar(
            'annealLoss',aloss,iters)
    log_writer.add_scalar(
            'totalLoss',loss,iters)
    log_writer.add_scalar(
            'psnr', psnr,iters)
    log_writer.add_scalar(
            'ssim', ssim,iters)


def train_model(outdir,model,optimizer,scheduler,\
        data_loader,datasize,device,num_epochs=25,itsz=1,sinter=5,pic='res',config=''):
    since          = time.time()
    iters = 0
    phase='train'
    #log_writer = SummaryWriter('./log')
    netLoss = lossFunc.totalLoss()

    try:
        for epoch in range(num_epochs):
            logging.info('-----Epoch {}-----'.format(epoch,num_epochs-1))
            t0 = time.time()
            epoch += 1
            model.train()
            iter_loss = 0.0
            psnr = 0.0
            ssim = 0.0
            it0 = time.time()
            for bims,bgts in data_loader[phase]:
                bims = bims.permute(0,3,1,2).to(torch.float32)
                bgts = bgts.permute(0,3,1,2).to(torch.float32)
                optimizer.zero_grad()
                preds,pred = model(bims)
                bloss,aloss = netLoss(
                        pred,preds,bgts,iters)
                loss = bloss+aloss
                #write_log(bloss,aloss,loss,psnr,ssim,iters,log_writer)
                loss.backward()
                optimizer.step()
                tloss = loss.item()
                iter_loss += tloss*(bims.size(0)*itsz)
                it1 = time.time()
                lr = scheduler.get_lr()
                _psnr = calculate_psnr(pred,bgts)
                _ssim = calculate_ssim(pred,bgts)
                psnr += _psnr
                ssim += _ssim
                if iters%1==0:
                    logging.info(\
                        'Epoch:{}/{}\tIters:{}\tLr:{}\tLoss:{:.6f}\tbasic:{:.6f}\tanneal:{:.6f}\ttime:{:.6f}s'\
                        .format(epoch,num_epochs,iters,lr[0]\
                        ,tloss,bloss.item(),aloss.item(),(it1-it0)%60))
                    logging.info('psnr/ssim:{:.6f}/{:6f}'.format(_psnr,_ssim))
                iters += 1
                it0 = it1
            if lr[0] > 1e-6:
                scheduler.step()
            epoch_loss = iter_loss / datasize[phase]
            psnr = psnr / datasize[phase]
            ssim = ssim / datasize[phase]
            t2 = time.time()
            eptime = t2-t0
            logging.info('epoch{}\tLoss:{:.6f}\tcost:{:.0f}m{:.0f}s'.format(
                epoch,epoch_loss,eptime//60,eptime%60))
            logging.info('avarage psnr/ssim:{:.6f}/{:.6f}'.format(psnr,ssim))
            if epoch%sinter==0:
                save_checkpoint(model,optimizer,outdir,epoch)
            save_compare_res(pred,bgts,bims,pic)
    except KeyboardInterrupt:
        logging.info('keyboard force terminate')
    finally:
        sname = os.path.join(outdir,'final_%d.pth'%iters)
        torch.save(model.state_dict(),sname)
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed//60,time_elapsed%60))
