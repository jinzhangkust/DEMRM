"""
Author: Dr. Jin Zhang
E-mail: j.zhang.vision@gmail.com
Created on 2024.05.24
"""

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import random
import shutil
import argparse
import numpy as np
from queue import Queue

from dataset import TailingSensorSet
from models.momentum_memory_network import Memory_Network
from util import AverageMeter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=60, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=850, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--resume', type=bool, default=True, help='restore checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--train_iteration', type=int, default=80, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--model_name', type=str, default='SemiDEMSM')  # WideResNet  InceptionV3

    # memory
    parser.add_argument('--mem_size', type=int, default=1500, help='number of memory slots')  # 2000
    parser.add_argument('--key_dim', type=int, default=128, help='key dimension')
    parser.add_argument('--val_dim', type=int, default=3, help='dimension of class distribution')
    parser.add_argument('--top_k_r', type=int, default=64, help='top_k for memory reading')  # 200
    parser.add_argument('--top_k_u', type=int, default=1, help='top_k for memory updating')  # 260
    parser.add_argument('--val_thres', type=float, default=0.08, help='threshold for value matching')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='threshold for value matching')  # 0.06
    parser.add_argument('--val_clip', type=float, default=0.4, help='threshold for value matching')
    parser.add_argument('--age_noise', type=float, default=8.0, help='number of training epochs')

    opt = parser.parse_args()

    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def set_loader(opt):
    full_data = TailingSensorSet(train_mode="train", transform="twice")
    train_size = int(0.7 * len(full_data))
    val_size = int(0.15 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))
    num_train_data = len(train_data)
    random.seed(42)
    train_index = list(range(num_train_data))
    random.shuffle(train_index)
    train_labeled_index = train_index[:int(num_train_data * 0.2)]
    train_unlabeled_index = train_index[int(num_train_data * 0.2):]

    train_labeled_sampler = torch.utils.data.SubsetRandomSampler(train_labeled_index)
    train_unlabeled_sampler = torch.utils.data.SubsetRandomSampler(train_unlabeled_index)

    train_labeled_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_labeled_sampler)
    train_unlabeled_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_unlabeled_sampler)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_labeled_loader, train_unlabeled_loader, val_loader, test_loader


def set_model(opt):
    model = Memory_Network(mem_size=opt.mem_size, key_dim=opt.key_dim, val_dim=opt.val_dim, top_k_r=opt.top_k_r)
    if opt.resume:
        resume = os.path.join(opt.save_folder, 'checkpoint_{}.ckpt'.format(opt.epoch))
        checkpoint = torch.load(resume)
        opt.epoch = checkpoint['epoch']
        model.encoder_q.load_state_dict(checkpoint['state_dict_query'])
        model.encoder_k.load_state_dict(checkpoint['state_dict_key'])
        key_np = checkpoint['key_slots']
        model.key_slots[0:key_np.shape[0], :] = torch.from_numpy(key_np).cuda()
        model.val_slots[0:key_np.shape[0], :] = torch.from_numpy(checkpoint['val_slots']).cuda()
        model.age[0:key_np.shape[0]] = torch.from_numpy(checkpoint['age_slots']).cuda()
        opt.val_thres = 0.06 #checkpoint['val_thres']

    criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        ce_criterion = ce_criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, ce_criterion


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.encoder_q.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
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


def train(labeled_loader, unlabeled_loader, model, criterion, ce_criterion, optimizer, epoch, opt, tb):
    model.train()
    val_thres_list = list()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse_losses = AverageMeter()
    entropy_losses = AverageMeter()
    contrast_losses = AverageMeter()
    con_losses = AverageMeter()
    entropy_losses_u = AverageMeter()
    mse_total_loss = 0
    entropy_total_loss = 0
    contrast_total_loss = 0
    con_total_loss = 0
    entropy_total_loss_u = 0
    warm_weight = 0.2 if epoch > 50 else 0
    end = time.time()
    labeled_train_iter = iter(labeled_loader)
    unlabeled_train_iter = iter(unlabeled_loader)
    for idx in range(opt.train_iteration):
        try:
            (_, reagents, images, targets) = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_loader)
            (_, reagents, images, targets) = next(labeled_train_iter)

        try:
            (_, reagents_u, images_u, _) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_loader)
            (_, reagents_u, images_u, _) = next(unlabeled_train_iter)
        data_time.update(time.time() - end)
        reagents = reagents.cuda(non_blocking=True)
        images = images[1].cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        reagents_u = reagents_u.cuda(non_blocking=True)
        images_u_w = images_u[0].cuda(non_blocking=True)
        images_u_s = images_u[1].cuda(non_blocking=True)
        bsz = targets.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        query, key = model(reagents.float(), images)
        pred, score = model.predict(query)
        mse_loss = criterion(pred, targets.float())
        entropy_loss = (-score * torch.log(score)).sum(dim=1).mean()
        contrast_loss = model.contrast_loss(query, targets, opt.val_thres)
        suploss = mse_loss + 0.1*entropy_loss + 0.05*contrast_loss

        query_u, _ = model(reagents_u.float(), images_u_s)
        pred_u, score_u = model.predict(query_u)
        with torch.no_grad():
            query_t_u, key_t_u = model(reagents_u.float(), images_u_w)
            pred_t_u, score_t_u = model.predict(key_t_u)
        con_loss = ce_criterion(score_u, score_t_u)
        entropy_loss_u = (-score_u * torch.log(score_u)).sum(dim=1).mean()
        unsuploss = 2*con_loss + 0.1*entropy_loss_u

        loss = suploss + warm_weight * unsuploss

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.key_enc_update()
            #query, key = model(reagents.float(), images)
            val_thres_temp = model.memory_update(query, key, targets, opt.val_thres, opt.val_ratio, update_weight=0.5)
            val_thres_list.append(val_thres_temp.cpu().numpy())
            #query_u, key_u = model(reagents_u.float(), images_u_w)
            if epoch > 50:
                val_thres_temp = model.memory_update(query_u, key_t_u, pred_t_u, opt.val_thres, opt.val_ratio, update_weight=0.1)
                val_thres_list.append(val_thres_temp.cpu().numpy())

        # update metric
        if idx:
            predict_set = np.append(predict_set, pred.detach().cpu().numpy(), axis=0)
            target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
        else:
            predict_set = pred.detach().cpu().numpy()
            target_set = targets.cpu().numpy()
        acc0, acc1, acc2, _, _, _, _, _, _ = cal_metrics(predict_set, target_set)

        mse_losses.update(mse_loss.item(), bsz)
        entropy_losses.update(entropy_loss.item(), bsz)
        contrast_losses.update(contrast_loss.item(), bsz)
        con_losses.update(con_loss.item(), bsz)
        entropy_losses_u.update(entropy_loss_u.item(), bsz)

        mse_total_loss += mse_loss.item()
        entropy_total_loss += entropy_loss.item()
        contrast_total_loss += contrast_loss.item()
        con_total_loss += con_loss.item()
        entropy_total_loss_u += entropy_loss_u.item()

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
                epoch, idx + 1, len(unlabeled_loader), batch_time=batch_time,
                data_time=data_time, loss=mse_losses, acc=acc0))
            sys.stdout.flush()

    # grid = torchvision.utils.make_grid(images)
    # tb.add_image("images", grid)
    # tb.add_graph(model, (tailings.float(),images))

    tb.add_scalar("Acc0", acc0, epoch)
    tb.add_scalar("Acc1", acc1, epoch)
    tb.add_scalar("Acc2", acc2, epoch)
    tb.add_scalar("Train/LabeledMSELoss", mse_total_loss, epoch)
    tb.add_scalar("Train/LabeledEntropyLoss", entropy_total_loss, epoch)
    tb.add_scalar("Train/LabeledContrastLoss", contrast_total_loss, epoch)
    tb.add_scalar("Train/UnlabeledConLoss", con_total_loss, epoch)
    tb.add_scalar("Train/UnlabeledEntropyLoss", entropy_total_loss_u, epoch)

    val_thres_epoch_mean = np.mean(val_thres_list)
    val_thres_epoch_max = np.max(val_thres_list)
    return val_thres_epoch_mean, val_thres_epoch_max


def validate(val_loader, model, criterion, epoch, opt, tb):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    mse_losses = AverageMeter()
    contrast_losses = AverageMeter()

    mse_total_loss = 0
    contrast_total_loss = 0

    with torch.no_grad():
        end = time.time()
        for idx, (_, reagents, images, targets) in enumerate(val_loader):
            reagents = reagents.cuda(non_blocking=True)
            images = images[0].cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]

            # forward
            query, key = model(reagents.float(), images)
            pred, score = model.predict(query)
            mse_loss = criterion(pred, targets.float())
            contrast_loss = model.contrast_loss(query, targets, opt.val_thres)

            # update metric
            if idx:
                predict_set = np.append(predict_set, pred.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            else:
                predict_set = pred.detach().cpu().numpy()
                target_set = targets.cpu().numpy()

            mse_losses.update(mse_loss.item(), bsz)
            contrast_losses.update(contrast_loss.item(), bsz)

            mse_total_loss += mse_loss.item()
            contrast_total_loss += contrast_loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'mse-loss {mseloss.val:.3f} ({mseloss.avg:.3f})\t'
                      'contrast-loss {contloss.val:.3f} ({contloss.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    mseloss=mse_losses, contloss=contrast_losses))

    acc0, acc1, acc2, _, _, _, _, _, _ = cal_metrics(predict_set, target_set)
    tb.add_scalar("Test-Acc0", acc0, epoch)
    tb.add_scalar("Test-Acc1", acc1, epoch)
    tb.add_scalar("Test-Acc2", acc2, epoch)
    tb.add_scalar("Test-MSE-Loss", mse_total_loss, epoch)
    tb.add_scalar("Test-Contrast-Loss", contrast_total_loss, epoch)
    return acc0, acc1, acc2


def main():
    best_acc0 = 0
    opt = parse_option()
    tb = SummaryWriter(comment="SemiDEMSM")

    train_labeled_loader, train_unlabeled_loader, val_loader, test_loader = set_loader(opt)
    model, criterion, ce_criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    opt.train_iteration = len(train_unlabeled_loader)
    print(f"len(train_unlabeled_loader): {len(train_unlabeled_loader)}   train_iteration: {opt.train_iteration}")
    val_thres_queue = MovingMax(10)
    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        val_thres_epoch_mean, val_thres_epoch_max = train(train_labeled_loader, train_unlabeled_loader, model, criterion, ce_criterion, optimizer, epoch, opt, tb)
        opt.val_thres = min(val_thres_epoch_mean, opt.val_thres)
        opt.val_clip = val_thres_queue.next(val_thres_epoch_mean)
        opt.val_clip = torch.from_numpy(np.array(opt.val_clip+0.02)).cuda()
        opt.val_thres = opt.val_thres + cosine_rampdown(epoch, ranmpdown_length=10)
        opt.val_thres = float(torch.clip(torch.from_numpy(np.array(opt.val_thres)), 0.006, opt.val_clip))
        print(f"epoch: {epoch}   val_clip: {opt.val_clip}   val_thres: {opt.val_thres}   val_thres_epoch_mean: {val_thres_epoch_mean}")

        acc0, acc1, acc2 = validate(val_loader, model, criterion, epoch, opt, tb)

        is_best = acc0 > best_acc0
        best_acc0 = max(acc0, best_acc0)
        if epoch % opt.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_query': model.encoder_q.state_dict(),
                'state_dict_key': model.encoder_k.state_dict(),
                'key_slots': model.key_slots.cpu().numpy(),
                'val_slots': model.val_slots.cpu().numpy(),
                'age_slots': model.age.cpu().numpy(),
                'val_thres': opt.val_thres,
                'optimizer': optimizer.state_dict(),
                'is_best': is_best,
            }, is_best, opt.save_folder, epoch)

def save_checkpoint(state, is_best, dirpath, epoch):
    if is_best:
        filename = 'best.ckpt'.format(epoch)
    else:
        filename = 'checkpoint_{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

def cosine_rampdown(epoch, ranmpdown_length = 10):
    state = epoch % ranmpdown_length  # 0（+）, 1（+）, 2（+）, 3（+）, 4（+）, 5（0）, 6（-）, 7（-）, 8（-）, 9（-）
    state = state if state < ranmpdown_length / 2 else state+1 # 0（+）, 1（+）, 2（+）, 3（+）, 4（+）, 5（-）, 6（-）, 7（-）, 8（-）, 9（-）
    out = 0.004 * float(np.cos(np.pi * state / ranmpdown_length))
    return out

class MovingMax():
    def __init__(self, size:int):
        self.que = Queue(size)
    def next(self, val: float) -> float:
        if self.que.full():
            self.que.get()
        self.que.put(val)
        return max(self.que.queue)

if __name__ == '__main__':
    main()
