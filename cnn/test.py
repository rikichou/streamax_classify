import os

import torch
from model import BaseNet, VGGNet_new

MODEL_PATH = r''
ckpt = r''

model = VGGNet_new(inplanes=3, num_classes=4).cuda()
model.parameters()

# state_dict = torch.load(ckpt, map_location='cpu')
# model.load_state_dict(state_dict, strict = False)

state_dict = torch.load(ckpt)
model.load_state_dict(state_dict)


print(model.parameters())
