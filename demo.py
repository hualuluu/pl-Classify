import os, cv2
import torch
import torchvision 
from PIL import Image
import numpy as np

imagepath = '/media/worker/98f56e45-a9c8-4f4e-b3f1-08a2d16a7ec1/liliang/Project/Classify/datasets/helmet/helmet/9e6a0cac9cd55f1f509c66dd31ef334d_0.jpg'
model_path = '/media/worker/98f56e45-a9c8-4f4e-b3f1-08a2d16a7ec1/liliang/Project/Classify/runs/20220802/weights/version_6/checkpoints/epoch=13-val_loss=0.00-other_metric=0.00.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = cv2.imread(imagepath)
image = cv2.resize(image, (256, 256))

image = Image.fromarray(image)

trans = torchvision.transforms.ToTensor()
image = trans(image).unsqueeze(0)

model = torchvision.models.resnet18(pretrained=False, num_classes=2)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = torch.nn.Identity()

# model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
pretrained_dict = torch.load(model_path, map_location=device)['state_dict']
new_pretrained_dict = {}
for k, v in pretrained_dict.items():
    if k.split('model.')[-1] in model.state_dict():
        # print(k, k.split('model.')[-1])
        new_k = k.split('model.')[-1]
        new_pretrained_dict[new_k] = v
model.load_state_dict(new_pretrained_dict)
model.eval()
outs = model(image)
print(outs)