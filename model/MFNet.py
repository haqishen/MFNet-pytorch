# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


class MFNet(nn.Module):

    def __init__(self, n_class):
        super(MFNet, self).__init__()
        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,16,16,36,36]

        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])

        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2]+inf_ch[2], rgb_ch[1]+inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0]+inf_ch[0], n_class)
        

    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:,:3]
        x_inf = x[:,3:]

        # encode
        x_rgb    = self.conv1_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2) # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)

        x_inf    = self.conv1_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2) # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)

        x = torch.cat((x_rgb, x_inf), dim=1) # fusion RGB and INF

        # decode
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool3
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        x = self.decode1(x)

        return x


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = MFNet(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,9,480,640), 'output shape (2,9,480,640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()
