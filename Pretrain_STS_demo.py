"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Created on 2023.04.27
"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import math
import argparse

from dataset import Data4MetricLearn
from models.DEFIE import defie
from util import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=30, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=371, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=9000, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # temperature
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # model
    parser.add_argument('--model_name', type=str, default='SSNetPreTrain')
    # creat argparse
    opt = parser.parse_args()
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def set_loader(opt):
    full_data = Data4MetricLearn()
    train_size = int(0.7 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    # Dataloader
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader

def set_model(opt):
    model = defie()
    if opt.epoch > 1:
        save_file = os.path.join(opt.save_folder, 'SSNetPreTrain_epoch_{epoch}.pth'.format(epoch=opt.epoch-1))
        model.load_state_dict(torch.load(save_file))
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion

def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer

def train(train_loader, model, criterion, optimizer, epoch, opt, tb):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (anchor_img, pos_img, neg_img) in enumerate(train_loader):
        data_time.update(time.time() - end)
        anchor_img = anchor_img.cuda(non_blocking=True)
        pos_img = pos_img.cuda(non_blocking=True)
        neg_img = neg_img.cuda(non_blocking=True)
        bsz = anchor_img.shape[0]
        # compute loss
        anchor_mapping = model(anchor_img)
        pos_mapping = model(pos_img)
        neg_mapping = model(neg_img)
        loss = criterion(anchor_mapping, pos_mapping, neg_mapping)
        # update metric
        losses.update(loss.item(), bsz)
        total_loss += loss.item()
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Debugging
        # for name, weight in model.named_parameters():
        #    if weight.requires_grad:
        #        print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if idx % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()
    tb.add_scalar("Loss", total_loss, epoch)
    return losses.avg

def validate(val_loader, model, criterion, epoch, opt, tb):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_loss = 0
    # predict_set = []
    # target_set = []

    end = time.time()
    with torch.no_grad():
        for idx, (anchor_img, pos_img, neg_img) in enumerate(val_loader):
            anchor_img = anchor_img.cuda(non_blocking=True)
            pos_img = pos_img.cuda(non_blocking=True)
            neg_img = neg_img.cuda(non_blocking=True)
            bsz = anchor_img.shape[0]
            # compute loss
            anchor_mapping = model(anchor_img)
            pos_mapping = model(pos_img)
            neg_mapping = model(neg_img)
            loss = criterion(anchor_mapping, pos_mapping, neg_mapping)
            # update metric
            losses.update(loss.item(), bsz)
            total_loss += loss.item()
            # print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx, len(val_loader), loss=losses))
                sys.stdout.flush()
    tb.add_scalar("Test-Loss", total_loss, epoch)

    return losses.avg


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="SSNetPreTrain")

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss_train = train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        loss_val = validate(val_loader, model, criterion, epoch, opt, tb)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'SSNetPreTrain_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
