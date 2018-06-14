# coding:utf-8
import numpy as np
from PIL import Image


def random_flip(image, label):
    if np.random.rand() < 0.5:
        image = image[:,::-1]
    return image


def random_crop(image, label, crop_rate=0.1, prob=1.0):
    
    if np.random.rand() < prob:

        w, h, c = image.shape

        h1 = np.random.randint(0, h*crop_rate)
        w1 = np.random.randint(0, w*crop_rate)
        h2 = np.random.randint(h-h*crop_rate, h+1)
        w2 = np.random.randint(w-w*crop_rate, w+1)

        image = image[w1:w2, h1:h2]
        label = label[w1:w2, h1:h2]

    return image, label
