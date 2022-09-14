import cv2
import torch
import torchvision 
from PIL import Image
import sys
sys.path.append('../')
from models.nets.net import Model
imagepath = '/media/worker/e5698782-da98-4582-a058-a26abe7eea03/Datasets/dz/select_cls/datasets/budaowei/0008_normal_frame_00000114_0.jpg'
model_path = '../runs/20220908_resnet18/version_0/weights/last.ckpt'
print(imagepath)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = cv2.imread(imagepath)
image = cv2.resize(image, (256, 256))

image = Image.fromarray(image)

trans = torchvision.transforms.ToTensor()
image = trans(image).unsqueeze(0)

m = Model('resnet18', num_classes = 3)
model = m.get_model()

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