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
from train import data_dir, model_dir

from tqdm import tqdm
from ipdb import set_trace as st


color = [
    [0,0,0],
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [255,255,255],
    [255,127,127]
]


def save_img(predictions, labels, names):

    for (i, name) in enumerate(names):
        out_pred  = np.zeros((640,480,3))
        out_label = np.zeros((640,480,3))

        for cid in range(1,9):
            out_pred[predictions[i]==cid] = color


def main():
    
    model = eval(args.model_name)(n_classes=9)

    train_dataset = MF_dataset(data_dir, 'train')
    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(train_dataset):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            logits = model(images)
            loss = softmax_with_cross_entropy(logits, labels)
            predictions = logits.argmax(1)
            acc = calculate_accuracy(predictions, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            save_img(predictions, labels, names)

            cur_t = time.time()
            print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                    % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))

            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test MFNet')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=16)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=80)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    main()