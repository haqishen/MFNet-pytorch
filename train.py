# coding:utf-8
import os
import argparse
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.augmentation import random_crop
from util.loss import softmax_with_cross_entropy
from util.MF_dataset import MF_dataset

from model import MFNet, SegNet

from tqdm import tqdm
from ipdb import set_trace as st

# config
data_dir  = '../../data/MF/'
model_dir = 'weights/'
augmentation_methods = []
lr_start  = 0.01
lr_decay  = 0.94
class_weight = None# torch.tensor([0.001, 0.02, 0.08, 0.1, 0.16, 0.21, 1., 0.525, 0.326])


def calculate_accuracy(predictions, labels):
    no_count = (labels==0).sum()
    count = ((predictions==labels)*(labels!=0)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc


def train(epo, model, train_loader, optimizer):

    lr_this_epo = lr_start * lr_decay**(epo-1)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr_this_epo

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        optimizer.zero_grad()
        logits = model(images)
        loss = softmax_with_cross_entropy(logits, labels, class_weight)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(1)
        acc = calculate_accuracy(predictions, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            t += 5

    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


def validation(epo, model, val_loader):
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            logits = model(images)
            loss = softmax_with_cross_entropy(logits, labels)
            predictions = logits.argmax(1)
            acc = calculate_accuracy(predictions, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            cur_t = time.time()
            print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                    % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))

    content = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' \
            % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


def main():

    os.makedirs(model_dir, exist_ok=True)
    model = eval(args.model_name)(n_classes=9)
    if args.gpu >= 0: model.cuda(args.gpu)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    if args.epoch_from > 1:
        print('| loading model from %s ...' % checkpoint_model_file)
        model.load_state_dict(torch.load(checkpoint_model_file, map_location={'cuda:0':'cuda:1'}))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))

    train_dataset = MF_dataset(data_dir, 'train', transform=augmentation_methods)
    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_dataset  = MF_dataset(data_dir, 'val')
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)

    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)


        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)

        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

    os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training MFNet')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=16)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=80)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir += args.model_name
    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    final_model_file      = os.path.join(model_dir, 'final.pth')
    log_file              = os.path.join(model_dir, 'log.txt')
    class_weight          = class_weight.cuda(args.gpu) if class_weight is not None else None

    print('| training %s on GPU #%d' % (args.model_name, args.gpu))
    print('| from epoch #%d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
