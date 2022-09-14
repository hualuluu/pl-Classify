import cv2
import torch
import torchvision 
from PIL import Image
import numpy as np
import sys
sys.path.append('../')
from models.nets.net import Model
from utils.utils_vis import get_cam
from torchvision import transforms
imagepath = '/media/worker/e5698782-da98-4582-a058-a26abe7eea03/Datasets/dz/select_cls/datasets/budaowei/0003_normal_frame_00000226_0.jpg'
model_path = '../runs/20220908_resnet18/version_0/weights/last.ckpt'
print(imagepath)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = cv2.imread(imagepath)
image = cv2.resize(image, (256, 256))
resize_image = image
image = Image.fromarray(image)

trans = transforms.Compose([
            transforms.ToTensor(),  # 归一化，像素值除以255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正规化，像素分布转换为同分布。这里的mean、std是imagenet上的均值标准差。
        ])
image = trans(image).unsqueeze(0)

# load model
m = Model('resnet18', num_classes = 3)
model = m.get_model()
# print(model)
# model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
pretrained_dict = torch.load(model_path, map_location=device)['state_dict']
new_pretrained_dict = {}
for k, v in pretrained_dict.items():
    if k.split('model.')[-1] in model.state_dict():
        # print(k, k.split('model.')[-1])
        new_k = k.split('model.')[-1]
        new_pretrained_dict[new_k] = v
model.load_state_dict(new_pretrained_dict)

# cam 
get_cam(resize_image, trans, model, 'layer4', './')

#output
model.eval()
outs = model(image)
print(outs)