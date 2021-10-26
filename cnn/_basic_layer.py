# ------------------------------------------------------------------------------
# driver fatigue clssificaiton
# Copyright (c) Streamax.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Hong Hu (huhong@streamax.com)
# baseline layer module from DSM multifacetask
# ------------------------------------------------------------------------------
import sys
import torch.nn as nn
import torch.nn.functional as F


# NOTE: mobilenet from MultiFaceTask in DSM
# __all__ = ['Conv2dBatchReLU', 'GlobalAvgPool2d']

class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, isPadding=True, isBias=False, summerName=None,
                 summerWrite=None):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.isBias = isBias
        self.summerWrite = summerWrite
        self.summerName = summerName
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)

        # Layer
        if isPadding == True:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                          bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 0, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)

        return x


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number."""

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        # print("gap", x.size(), x)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(B, C, 1, 1)

        return x

class GlobalMaxPool2d(nn.Module):
    """ This layer averages each channel to a single number."""

    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, x):
        # print("gap", x.size(), x)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.max_pool2d(x, (H, W))
        x = x.view(B, C, 1, 1)

        return x


class FullyConnectLayer(nn.Module):
    """ This layer averages each channel to a single number."""

    def __init__(self, in_channels, out_channels):
        super(FullyConnectLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )

    def forward(self, x):
        if len(x.size()) < 2:
            print("FullyConnectLayer input error!\n")
            sys.exit()
        flattenNum = 1
        for i in range(1, len(x.size())):
            flattenNum *= x.size(i)
        x = x.view(-1, flattenNum)
        x = self.layers(x)

        return x

class BasicConv(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y