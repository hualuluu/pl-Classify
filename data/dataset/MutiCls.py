#!/usr/bin/python3
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.utils import get_singleclass_txt

image_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

class MutiCls(Dataset):
    """
    多类别分类数据加载
    需要有txt标注文件： imagepath|cls
    """
 
    def __init__(self, root_dir, dims = (256, 256), train=True, transform=None):
        # dataset root dir
        self.root_dir = root_dir
        self.dims = dims
        if train:
            self.labelpath = self.root_dir + 'train.txt'
        else:
            self.labelpath = self.root_dir + 'test.txt'
        
        self.image_name, self.label = get_singleclass_txt(self.labelpath)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        
        # BGR
        image = cv2.imread(self.image_name[idx])
        
        image = cv2.resize(image, self.dims)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        label = self.label[idx]
        label = torch.from_numpy(np.array([label]))
        
        return image, label

if __name__ == '__main__':
    Clothescolordatasets = Helmet("/home/worker/ll/code/clothes_color_cls/data/clothescolor/", True, image_transform)
    trainloader = DataLoader(Clothescolordatasets, batch_size=8, shuffle=True)

    for i, data in enumerate(trainloader):
        image, label = data
        #print(image, label)