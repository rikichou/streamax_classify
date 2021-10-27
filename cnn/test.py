import os

import cv2
import numpy as np
import torch
from model import BaseNet, VGGNet_new

# model config
insize = 300
mean = (104, 117, 123)

# load model and parameters
ckpt = r'E:\workspace\pro\streamax_classify\cnn\models\1026\epoch_22.pth.tar'

model = VGGNet_new(inplanes=3, num_classes=4).cuda()
# for name, param in model.named_parameters():
#     print(name)
#     print(param)
model_saved = torch.load(ckpt, map_location='cpu')
state_dict = model_saved['state_dict']
model.load_state_dict(state_dict, strict=True)

# enter eval mode
model.eval()

# state_dict = torch.load(ckpt)
# model.load_state_dict(state_dict)

# test image
image_path = r'E:\workspace\pro\facialExpression\data\Selected\val\neutral\AffectNet_cd2acd5c39ae658b586a4baea6dc6d8abed554e7f8c6080482a96869.jpg'
image = cv2.imread(image_path)

image = cv2.resize(image, (insize, insize),interpolation=cv2.INTER_LINEAR)
image = image.astype(np.float32)
image -= mean
input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).cuda()

pred = model(input_tensor)

print(pred)
