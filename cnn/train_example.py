# ------------------------------------------------------------------------------
# driver fatigue clssificaiton
# Copyright (c) Streamax.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Hong Hu (huhong@streamax.com)
# train loop
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models

from model import BaseNet, VGGNet_new
from dataset import DatasetFacialExpression
from loss import FocalLossWithSigmoid
from torch.nn import CrossEntropyLoss
from utils import get_model_summary
from optimizer import SGD_GC
#from transforms import _ToTensor
import math

import logging
import pprint
from collections import OrderedDict


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser(description='DSM Fatigue training loop')
    parser.add_argument('train_ann_file', type=str,
                        help='train annotation file path')
    parser.add_argument('val_ann_file', type=str,
                        help='valid annotation file path')
    parser.add_argument('--net_type', default='basenet', type=str,
                        help='networktype: cnn, and lstm')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--num_classes', type=int, default=4, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--depth', default=32, type=int,
                        help='depth of the network (default: 32)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                        help='to use basicblock for CIFAR datasets (default: bottleneck)')
    parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                        help='dataset (options: cifar10, cifar100, and imagenet)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                        help='to print the status at every iteration')
    parser.add_argument('--alpha', default=300, type=float,
                        help='number of new channel increases per depth (default: 300)')
    parser.add_argument('--expname', default='TEST', type=str,
                        help='name of experiment')
    parser.add_argument('--beta', default=0, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                        help='cutmix probability')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(verbose=True)

    parser.add_argument('--opt_level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument("--init_checkpoint", default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    global args, best_prec1, best_epoch

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    args = parse()
    logger.info("opt_level = {}".format(args.opt_level))
    logger.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32))
    logger.info("loss_scale = {}".format(args.loss_scale))
    logger.info("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))
    logger.info(pprint.pformat(args))
    model_names = sorted(name for name in BaseNet.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(BaseNet.__dict__[name]))
    logger.info(model_names)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        DatasetFacialExpression(
            data_prefix='data',
            ann_file='train_ann_file',
            transform = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #_ToTensor(ones_input=False),
                normalize,
            ]),
            if_bgr=True,
            insize=300,
            train=True,
            mode='color',
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        DatasetFacialExpression(
            data_prefix='data',
            ann_file='train_ann_file',
            transform = transforms.Compose([
                transforms.Resize((300,300)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            if_bgr=True,
            insize=300,
            train=False,
            mode='color',
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    logger.info("=> creating model '{}'".format(args.net_type))
    
    model = VGGNet_new(inplanes=3, num_classes=args.num_classes).cuda()

    # if args.init_checkpoint != None:
    #     model.load_state_dict(torch.load(args.init_checkpoint)['state_dict'], strict= True)

    state_dict = torch.load(args.init_checkpoint, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if str(k).startswith('base'):
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict = False)


    dump_input = torch.rand((1, 3, 300, 300)).cuda()
    logger.info(get_model_summary(model, dump_input))

    criterion = CrossEntropyLoss().cuda()

    optimizer = SGD_GC(model.parameters(), 
                       args.lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       nesterov=True)

    best_prec1 = 0
    best_epoch = 0

    logger.info("############ Start training ############")
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(valid_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            is_best = True
            best_prec1 = prec1
            best_epoch = epoch + 1

        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # NOTE: check is_best
        is_best = False
        logger.info('=> best acc {:.4f} at epoch {}.'.format(best_prec1, best_epoch))

    logger.info('*' * 100)
    logger.info('=> best acc {:.4f} at epoch {}.'.format(best_prec1, best_epoch))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    wrongPred = {'0->1':0, '0->2':0, '1->0':0, '1->2':0, '2->0':0, '2->1':0}
    for input, target in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output= model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output.data, target, topk=(1, 2))
        wPred = accuracy_cal(output.data, target)
        for w in wPred:
            wrongPred[w] += wPred[w]

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top2.update(acc2.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('*Train Epoch: [{0}/{1}]\t Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} \tTrain Loss {loss.avg:.3f}'.format(
        epoch+1, args.epochs, top1=top1, top2=top2, loss=losses))
    print('Wrong Predictions:')
    print(wrongPred)

    return losses.avg

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    wrongPred = {'0->1':0, '0->2':0, '1->0':0, '1->2':0, '2->0':0, '2->1':0}
    for input, target in val_loader:
        target = target.cuda()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output.data, target, topk=(1, 2))
        wPred = accuracy_cal(output.data, target)
        for w in wPred:
            wrongPred[w] += wPred[w]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top2.update(acc2.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('*Valid Epoch: [{0}/{1}]\t Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} \t Test Loss {loss.avg:.3f}'.format(
        epoch+1, args.epochs, top1=top1, top2=top2, loss=losses))
    print('Wrong Predictions:')
    print(wrongPred)
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        save_name = 'runs/{}/epoch_{}_{:.4f}_model_best.pth.tar'.format(args.expname, state['epoch'], 
                                                                        state['best_prec1'])

        shutil.copyfile(filename, save_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    '''
    if epoch <= 30:
        lr = 1e-2
    elif epoch > 30 and epoch <= 40:
        lr = 1e-3
    elif epoch > 40 and epoch <= 50:
        lr = 1e-4
    else:
        lr = 1e-5
    '''
    # lr = args.lr * (0.1 ** ((epoch+1) // (args.epochs * 0.5))) * (0.1 ** ((epoch+1) // (args.epochs * 0.75)))
    # print('lr: ')
    # print(lr)
    #lr = 0.5 * (1 + math.cos((epoch+1)*math.pi/args.epochs)) * args.lr
    lr = args.lr
    # lr = 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_cal(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    pred = torch.argmax(output, axis=1)

    #wrongPred = {'0':0, '1':0, '2':0}
    wrongPred = {'0->1':0, '0->2':0, '1->0':0, '1->2':0, '2->0':0, '2->1':0}
    correct = pred.eq(target)

    for i, t in enumerate(correct):
        if t == False:
            if target[i].cpu().numpy() == 0:
                if pred[i].cpu().numpy() == 1:
                    wrongPred['0->1'] += 1
                else:
                    wrongPred['0->2'] += 1
            elif target[i].cpu().numpy() == 1:
                if pred[i].cpu().numpy() == 0:
                    wrongPred['1->0'] += 1
                else:
                    wrongPred['1->2'] += 1
            else:
                if pred[i].cpu().numpy() == 0:
                    wrongPred['2->0'] += 1
                else:
                    wrongPred['2->1'] += 1

    return wrongPred

if __name__ == '__main__':
    main()
