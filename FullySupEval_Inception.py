"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Created on 2024.04.17
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from dataset import TailingSensorSet
from models.inception_proj import inception

from util import AverageMeter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=150, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='FullySupLinearInception')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # create
    opt = parser.parse_args()
    # save
    opt.save_folder = os.path.join('save/', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class LinearSensor(nn.Module):
    def __init__(self, opt):
        super(LinearSensor, self).__init__()
        self.feature = inception()
        self.predictor = nn.Sequential(
            nn.Linear(2051, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x0, x1):
        x1 = self.feature(x1)
        x = torch.cat((x0, x1.view(x1.size(0), -1)), dim=1)
        x = self.predictor(x)
        return x


def set_loader(opt):
    full_data = TailingSensorSet(train_mode="train")
    train_size = int(0.7 * len(full_data))
    val_size = int(0.15 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = LinearSensor(opt)
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer

def cal_metrics(y_pred, y_true):
    # r2_score: R2
    R2_0 = r2_score(y_true[:, 0], y_pred[:, 0])
    R2_1 = r2_score(y_true[:, 1], y_pred[:, 1])
    R2_2 = r2_score(y_true[:, 2], y_pred[:, 2])
    # mean_squared_error (If True returns MSE value, if False returns RMSE value.): RMSE
    RMSE_0 = mean_squared_error(y_true[:, 0], y_pred[:, 0], squared=False)
    RMSE_1 = mean_squared_error(y_true[:, 1], y_pred[:, 1], squared=False)
    RMSE_2 = mean_squared_error(y_true[:, 2], y_pred[:, 2], squared=False)
    # mean_absolute_error: MAE
    MAE_0 = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    MAE_1 = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    MAE_2 = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    return R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2

def warmup_learning_rate(opt, epoch, idx, nBatch, optimizer):
    T_total = opt.epochs * nBatch
    T_warmup = 10 * nBatch
    if epoch <= 10 and idx <= T_warmup:
        lr = 1e-6 + (opt.learning_rate - 1e-6) * idx / T_warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, opt, tb):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, reagents, images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        reagents = reagents.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]
        # compute loss
        output = model(reagents.float(), images)
        loss = criterion(output, targets.float())
        # update metric
        if idx == 0:
            predict_set = output.detach().cpu().numpy()
            target_set = targets.cpu().numpy()
        else:
            predict_set = np.append(predict_set, output.detach().cpu().numpy(), axis=0)
            target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
        losses.update(loss.item(), bsz)
        total_loss += loss.item()
        R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_set, target_set)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'R-square {acc:.3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=R2_0))
            sys.stdout.flush()
    # tensorboard
    tb.add_scalar("Acc0", R2_0, epoch)
    tb.add_scalar("Acc1", R2_1, epoch)
    tb.add_scalar("Acc2", R2_2, epoch)
    tb.add_scalar("Loss", total_loss, epoch)
    return losses.avg


def validate(val_loader, model, criterion, epoch, opt, tb):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_loss = 0
    with torch.no_grad():
        end = time.time()
        for idx, (_, reagents, images, targets) in enumerate(val_loader):
            reagents = reagents.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]
            # forward
            output = model(reagents.float(), images)
            loss = criterion(output, targets.float())
            if idx:
                predict_set = np.append(predict_set, output.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            else:
                predict_set = output.detach().cpu().numpy()
                target_set = targets.cpu().numpy()
            # update metric
            losses.update(loss.item(), bsz)
            total_loss += loss.item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses))
    # tensorboard
    R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_set, target_set)
    tb.add_scalar("Test-Acc0", R2_0, epoch)
    tb.add_scalar("Test-Acc1", R2_1, epoch)
    tb.add_scalar("Test-Acc2", R2_2, epoch)
    tb.add_scalar("Test-Loss", total_loss, epoch)

    if epoch >= opt.total_epochs-10:
        R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_set, target_set)
        print('R2_0: %.4f, R2_1: %.4f, R2_2: %.4f, RMSE_0: %.4f, RMSE_1: %.4f, RMSE_2: %.4f, MAE_0: %.4f, MAE_1: %.4f, MAE_2: %.4f' % (
            R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2))
    return losses.avg


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="FullySupLinearInception")

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss_train = train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        loss_val = validate(val_loader, model, criterion, epoch, opt, tb)
        print('epoch {}, total time {:.2f}, loss_train {}, loss_val {}'.format(epoch, time2 - time1, loss_train, loss_val))
        if epoch % opt.save_freq == 0 and epoch >= 100:
            save_file = os.path.join(opt.save_folder, 'FullySupLinearInception_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
