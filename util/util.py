# coding:utf-8
import numpy as np
import chainer
from PIL import Image
from ipdb import set_trace as st

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc


def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    IoU = np.zeros(n_class)
    conf[:,0] = cf[:,0]/cf[:,0].sum()
    for cid in range(1,n_class):
        if cf[:,cid].sum() > 0:
            conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
            IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU


# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(names, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(names[i].replace('.png', '_pred.png'))
