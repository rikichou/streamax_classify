# ------------------------------------------------------------------------------
# driver fatigue clssificaiton
# Copyright (c) Streamax.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Hong Hu (huhong@streamax.com)
# loss function
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLossWithSigmoid(nn.Module):
    def __init__(self, numClass, gamma, alpha):
        super(FocalLossWithSigmoid, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

    def forward(self, inputs, target):
        # inputs.size()  B*C
        # target.size()  B
        N = inputs.size(0)  # batch size
        C = inputs.size(1)  # class num
        if np.any(np.isnan(inputs.cpu().detach().numpy())):
            print("inputs1 has nan", inputs)
        dtype = target.dtype
        device = target.device

        t = target.unsqueeze(1)
        classRange = torch.arange(0, C, dtype=dtype, device=device).unsqueeze(0)

        posMask = (t == classRange)
        negMask = (t != classRange)

        sigmoidOut = torch.sigmoid(inputs)

        if np.any(np.isnan(inputs.cpu().detach().numpy())):
            print("inputs2s has nan", inputs)

        if np.any(np.isnan(sigmoidOut.cpu().detach().numpy())):
            print("sigmoidOut has nan", sigmoidOut)
        posSigmoidOut = (1 - sigmoidOut) ** self.gamma * torch.log(torch.clamp(sigmoidOut, 1e-6))
        negSigmoidOut = sigmoidOut ** self.gamma * torch.log(torch.clamp(1 - sigmoidOut, 1e-6))
        focalLossOut = -posMask.float() * posSigmoidOut * self.alpha - \
                       negMask.float() * negSigmoidOut * (1 - self.alpha)
        # print("posGigmoidOut", posSigmoidOut)
        # print("negSigmodiOut", negSigmoidOut)
        if np.any(np.isnan(inputs.cpu().detach().numpy())):
            print("input has nan", inputs)

        if np.any(np.isnan(target.cpu().detach().numpy())):
            print("target has nan")

        if np.any(np.isnan(negSigmoidOut.cpu().detach().numpy())):
            print("negSigmoidOut has nan")

        if np.any(np.isnan(posSigmoidOut.cpu().detach().numpy())):
            print("posSigmoidOut has nan")
            sys.exit()

        return focalLossOut.sum() / N
