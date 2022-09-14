import torchvision 
import torch

def ResNet18(num_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    
    return model
