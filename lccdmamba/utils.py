import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, padding=0, bias=True):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        y = self.conv2(x)
        y = self.bn(y)
        return y

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, padding=0, bias=True, act='relu',channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        if act == "silu":
            self.act = nn.SiLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'softmax':
            self.act = nn.Softmax()
        elif act == "sigmoid":
            self.act == nn.Sigmoid()
    
    def forward(self, input):
        if self.channel_first:
            x = input
        else:
            x = input.permute(0, 3, 1, 2)
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        if self.channel_first:
            return y
        y = y.permute(0, 2, 3, 1)
        return y


class BNAct(nn.Module):
    def __init__(self, channels, act='relu'):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        if act == "silu":
            self.act = nn.SiLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'softmax':
            self.act = nn.Softmax()
        elif act == "sigmoid":
            self.act == nn.Sigmoid()
    
    def forward(self, x):
        y = self.bn(x)
        y = self.act(y)
        return y

class DecomposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super(DecomposedConv, self).__init__()
        pad = int(kernels // 2)
        self.conv_vert = nn.Conv2d(in_channels, out_channels, kernel_size=(kernels, 1), padding=(pad, 0))
        self.conv_horiz = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernels), padding=(0, pad))

    def forward(self, x):
        x = self.conv_vert(x)
        x = self.conv_horiz(x)
        return x

