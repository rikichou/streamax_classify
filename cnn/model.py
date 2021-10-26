# ------------------------------------------------------------------------------
# driver fatigue clssificaiton
# Copyright (c) Streamax.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Hong Hu (huhong@streamax.com)
# simple baseline model
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn

# from ._basic_layer import Conv2dBatchReLU, GlobalAvgPool2d, FullyConnectLayer
from _basic_layer import Conv2dBatchReLU, GlobalAvgPool2d, FullyConnectLayer, BasicConv, CALayer, GlobalMaxPool2d

import logging

logger = logging.getLogger(__name__)


class BaseNet(nn.Module):
    def __init__(self, inplanes, num_classes=2):
        super().__init__()
        self.pre_layers = nn.Sequential(
            #PreAttention(512, 0.1, 2),
            Conv2dBatchReLU(inplanes, 32, 3, (1, 2)),  # 30, 256
            Conv2dBatchReLU(32, 64, 3, (1, 2)),  # 30, 128
            #OutputAttention(30 * 128, 64, 128, 0.1), # 1
            Conv2dBatchReLU(64, 64, 3, (1, 2)),  # 30, 64
            Conv2dBatchReLU(64, 128, 3, (1, 1)),
            #OutputAttention(30 * 64, 128, 64, 0.1), # 2
            Conv2dBatchReLU(128, 128, 3, (1, 2)),  # 30,32
            Conv2dBatchReLU(128, 128, 3, (1, 1)),
            #OutputAttention(30 * 32, 128, 32, 0.1), #3
            Conv2dBatchReLU(128, 256, 3, (2, 2)),  # 15,16
            OutputAttention(15 * 16, 256, 16, 0.1), #4
            Conv2dBatchReLU(256, 512, 3, (2, 2))  # 8,8
        )

        self.pool = GlobalAvgPool2d()
        self.classifier = FullyConnectLayer(512, num_classes)
        def init_weights(module):
            #logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # if os.path.isfile(pretrained):
            #     pretrained_dict = torch.load(pretrained)
            #     logger.info('=> loading pretrained model {}'.format(pretrained))
            #     model_dict = self.state_dict()
            #     pretrained_dict = {k: v for k, v in pretrained_dict.items()
            #                     if k in model_dict.keys()}
            #     for k, _ in pretrained_dict.items():
            #         logger.info('=> loading {} pretrained model {}'.format(k, pretrained))
            #     model_dict.update(pretrained_dict)
            #     self.load_state_dict(model_dict)
        self.apply(init_weights)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

base_channel = 16

# 2,763,043
class VGGNet(nn.Module):
    def __init__(self, inplanes = 3, num_classes = 10):
        super().__init__()

        self.stage1 = nn.Sequential(
            BasicConv(inplanes, base_channel, kernel_size=3, stride=1, padding=1),
            BasicConv(base_channel, base_channel, kernel_size=3, stride=1, padding=1), #32 * 32
            BasicConv(base_channel, base_channel * 2, kernel_size=3, stride=1, padding=1)
        )

        self.stage2 = nn.Sequential(
            BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1), #16 * 16
            BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        )
        
        self.stage3 = nn.Sequential(
            BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1), #8 * 8
            BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)
        )

        self.stage4 = nn.Sequential(
            BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1),# 4 * 4
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)
        )

        self.stage5 = nn.Sequential(
            BasicConv(base_channel * 8, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 128, kernel_size=3, stride=2, padding=1) # 2 * 2
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        y = self.stage5(x)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                 [2:]).view(y.size(0), -1)

        x = self.classifier(y)
        return x

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGGNet_new(nn.Module):
    def __init__(self, inplanes = 3, num_classes = 10):
        super().__init__()

        self.base = nn.ModuleList(VGG())

        self.avg_pool = GlobalAvgPool2d()
        
        self.classifier = FullyConnectLayer(128, num_classes)

    def forward(self, x):
        
        for k in range(len(self.base)):
            x = self.base[k](x)

        x = self.avg_pool(x)
        x = self.classifier(x.view(x.size(0), -1))

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def VGG():
    layers = []
    layers += [BasicConv(3, base_channel)]
    layers += [BasicConv(base_channel, base_channel, kernel_size=3,stride=2, padding=1)] #150 * 150

    layers += [BasicConv(base_channel, base_channel * 2, kernel_size=3,stride=1, padding=1)]
    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1)] #75 * 75

    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1)] #38 * 38

    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)] # 19 * 19

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8,128,kernel_size=1,stride=1,padding=0)]
    layers += [BasicConv(128,128,kernel_size=3,stride=2, padding=1)] # 10*10

    return layers

def get_cls_net(config, **kwargs):
    model = BaseNet(inplanes=3, num_classes=2)
    model.init_weights()
    return model

class Rotation_Cls(nn.Module):

    def __init__(self, num_classes = 10):
        super().__init__()
        # self.conv = nn.Sequential(
        #     Conv2dBatchReLU(3, 16, 3, 1), # 64 
        #     Conv2dBatchReLU(16, 32, 3, 2), # 32
        #     Conv2dBatchReLU(32, 64, 1, 1), # 32
        #     Conv2dBatchReLU(64, 64, 3, 2), # 16
        #     Conv2dBatchReLU(64, 128, 1, 1), # 16
        #     Conv2dBatchReLU(128, 128, 3, 2) # 8
        # )
        self.conv = nn.Sequential(
            Conv2dBatchReLU(3, 8, 3, 1), # 64 
            Conv2dBatchReLU(8, 16, 3, 2), # 32
            Conv2dBatchReLU(16, 32, 1, 1), # 32
            Conv2dBatchReLU(32, 32, 3, 2), # 16
            Conv2dBatchReLU(32, 64, 1, 1), # 16
            Conv2dBatchReLU(64, 128, 3, 2) # 8
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=(4,4))
        x = F.avg_pool2d(x, kernel_size=(2,2))
        x = self.classifier(x.view(x.size(0), -1))

        return x


if __name__ == "__main__":
    model = BaseNet(inplanes=1, num_classes=2).cuda()
    input = torch.ones(1, 1, 32, 512).cuda()
    output = model(input)
    print(output.shape)
