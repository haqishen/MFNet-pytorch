# MFNet-pytorch

this is pytorch implementation of [MFNet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes](https://ieeexplore.ieee.org/document/8206396/) (IROS 2017). The pdf can be downloaded from [HERE](https://drive.google.com/file/d/1vxMh63QpdxPnG3jhzpQU0fb-2XOzHR-Z/view?usp=sharing) (Google Drive shared file). 

If we upload this paper to `arxiv.org` in the first time, the number should be something like `1612.xxxx`, but we forgot to do it! Nevertheless, we don't want it look very new (like `1806.xxxx`), so... please download it from google driver if you need it.

## Introduction

MFNet is a light CNN architecture for multispectral images semantic segmentation, with ~ 1/40x parameters and 6x ~ inference speed, while providing similar or higher accuracy compared to SegNet.

## Requirements

* pytorch 0.4.0
* PIL 4.3.0
* numpy 1.14.0
* tqdm 4.19.4

## Usage
