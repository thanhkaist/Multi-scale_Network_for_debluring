import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

# Model related functions

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

def no_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Super Resolution
class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_suffle = nn.PixelShuffle(up_scale)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_suffle(x)
        x = self.tanh(x)
        return x



class OneScale(nn.Module):
    def __init__(self,input_channel=3,long_skip = True):
        super(OneScale,self).__init__()

        self.long_skip = long_skip
        self.input_channels = input_channel

        self.block1 = nn.Sequential(
            nn.Conv2d(self.input_channels,64,kernel_size=5,padding=5//2),
            nn.ReLU())

        block2 = [ResidualBlock(64) for _ in range(9)]
        self.block2 = nn.Sequential(*block2)

        self.block3 = nn.Conv2d(64,3,kernel_size=5,padding=5//2)

    def forward(self, img):
        out1 = self.block1(img)
        out2 = self.block2(out1)
        if self.long_skip:
            out3 = self.block3(out1+out2)
        else:
            out3 = self.block3(out2)
        return out3



class MultiScale(nn.Module):
    def __init__(self,long_skip = True):
        super(MultiScale,self).__init__()

        self.long_skip = long_skip

        self.branch256 = OneScale(6,self.long_skip)
        self.branch128 = OneScale(6,self.long_skip)
        self.upconv128_256 = UpsampleBlock(3,2)
        self.branch64 = OneScale(3,self.long_skip)
        self.upconv64_128 = UpsampleBlock(3, 2)

    def forward(self, blur256,blur128,blur64):
        out1 = self.branch64(blur64)
        up1 = self.upconv64_128(out1)
        out2 = self.branch128(torch.cat((out1,up1),1))
        up2 = self.upconv128_256(out1)
        out3 = self.branch256(torch.cat((out2,up2),1))
        return (out3,out2,out1)






class ResidualBlock(nn.Module):

    def __init__(self, channels = 64, scale_factor = 1):
        super(ResidualBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual.mul(self.scale_factor)