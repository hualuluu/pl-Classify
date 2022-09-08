import torchvision 
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    
    def __init__(self, num_classes=10):
    
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=2)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = torch.nn.Identity()
    
    def forward(self, x):
        y = self.model(x)
        return y