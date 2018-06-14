# coding:utf-8
import numpy as np
from PIL import Image


def random_flip(image, label):
    if np.random.rand() < 0.5:
        image = image[:,::-1]
    return image


class RandomCrop(object):
    def __init__(self, crop_rate=0.1, prob=1.0):
        super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label
