# coding:utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as st

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Encoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_channels,  out_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )
    def forward(self,x):
        x      = self.encode(x)
        x_down = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_down


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True):
        super(MiniInception, self).__init__()
        self.is_down = is_down
        self.conv1_left  = ConvBnRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)

    def forward(self,x):
        x      = torch.cat([self.conv1_left(x), self.conv1_right(x)], 1)
        x      = torch.cat([self.conv2_left(x), self.conv2_right(x)], 1)
        x      = torch.cat([self.conv3_left(x), self.conv3_right(x)], 1)
        if self.is_down:
            x_down = F.max_pool2d(x, kernel_size=2, stride=2)
            return x, x_down
        else:
            return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv=ConvBnRelu2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
    def forward(self, x_big, x):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x+x_big)
        return  x


class MFNet(nn.Module):
    def __init__(self, n_classes):
        super(MFNet, self).__init__()
        # self.n_classes  = n_classes
        # self.n_channels = n_channels
        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,16,16,36,36]

        self.down1_rgb = ConvBnRelu2d(3, rgb_ch[0])
        self.down2_rgb = Encoder(rgb_ch[0], rgb_ch[1])
        self.down3_rgb = Encoder(rgb_ch[1], rgb_ch[2])
        self.down4_rgb = MiniInception(rgb_ch[2], rgb_ch[3])
        self.center_rgb = MiniInception(rgb_ch[3], rgb_ch[4], is_down=False)

        self.down1_inf = ConvBnRelu2d(1, inf_ch[0])
        self.down2_inf = Encoder(inf_ch[0], inf_ch[1])
        self.down3_inf = Encoder(inf_ch[1], inf_ch[2])
        self.down4_inf = MiniInception(inf_ch[2], inf_ch[3])
        self.center_inf = MiniInception(inf_ch[3], inf_ch[4], is_down=False)

        self.up3 = Decoder(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.up2 = Decoder(rgb_ch[2]+inf_ch[2], rgb_ch[1]+inf_ch[1])
        self.up1 = Decoder(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.classify = ConvBnRelu2d(rgb_ch[0]+inf_ch[0], n_classes)


    def forward(self, x):
        x_rgb = x[:,:3]
        x_inf = x[:,3:]

        x_rgb           = F.max_pool2d(self.down1_rgb(x_rgb), kernel_size=2, stride=2)
        tmp2_rgb, x_rgb = self.down2_rgb(x_rgb)
        tmp3_rgb, x_rgb = self.down3_rgb(x_rgb)
        tmp4_rgb, x_rgb = self.down4_rgb(x_rgb)
        x_rgb           = self.center_rgb(x_rgb)

        x_inf           = F.max_pool2d(self.down1_inf(x_inf), kernel_size=2, stride=2)
        tmp2_inf, x_inf = self.down2_inf(x_inf)
        tmp3_inf, x_inf = self.down3_inf(x_inf)
        tmp4_inf, x_inf = self.down4_inf(x_inf)
        x_inf           = self.center_inf(x_inf) 

        x = torch.cat([x_rgb, x_inf], 1)

        x = self.up3(torch.cat([tmp4_rgb, tmp4_inf], 1), x)
        x = self.up2(torch.cat([tmp3_rgb, tmp3_inf], 1), x)
        x = self.up1(torch.cat([tmp2_rgb, tmp2_inf], 1), x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.classify(x)

        return x


