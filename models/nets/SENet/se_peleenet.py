import torch
import torch.nn as nn

import sys
sys.path.append(".") 
from .se_module import SELayer
from ..DenseNet.csppeleenet import PeleeNet, Conv_bn_relu

class SETwoDenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growthRate, reduction = 16):
        super(SETwoDenseBlock, self).__init__()

        self.tdp1_a = Conv_bn_relu(inp, inter_channel,1,1,0)
        self.tdp1_b = Conv_bn_relu(inter_channel, growthRate, 3,1,1)
        self.left = nn.Sequential(self.tdp1_a,
                                  self.tdp1_b
                                )

        self.tdp2_a = Conv_bn_relu(inp, inter_channel, 1,1,0)
        self.tdp2_b = Conv_bn_relu(inter_channel, growthRate,3,1,1)
        self.tdp2_c = Conv_bn_relu(growthRate, growthRate,3,1,1)
        self.right = nn.Sequential(self.tdp2_a,
                                   self.tdp2_b,
                                   self.tdp2_c
                                )

        self.se_inp = growthRate * 2 + inp
        self.se = SELayer(self.se_inp , reduction)
    def forward(self, x):
        left_feat = self.left(x)
        
        right_feat = self.right(x)

        cat_feat = torch.cat([x,left_feat,right_feat],1)
        #print(cat_feat.shape)
        se_feat = self.se(cat_feat)
        return se_feat

def sepeleenet(model_name, class_num):
    model_dict = {
        'sepeleenet_csp':PeleeNet(num_classes = class_num, partial_ratio = 0.5, TwoDenseBlock = SETwoDenseBlock),
        'sepeleenet':PeleeNet(num_classes = class_num, partial_ratio = 1.0, TwoDenseBlock = SETwoDenseBlock)
    }
    model = model_dict[model_name].cuda()
    
    return model

def CAM_CSPFeatures(input, model_path):
    model = torch.load(model_path).cpu()
    print(model)
    model.eval()
    x = model.stem(input)
    feature = model.stages(x)
    # global average pooling layer
    pool_feature = model.pool(feature)
        
    fc = model.classifier(pool_feature.reshape(pool_feature.shape[0], -1))
    return feature, fc

if __name__ == '__main__':
    p = PeleeNet(num_classes=1000)
    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224))
    output = p(input)

    print(output.size())

    # torch.save(p.state_dict(), 'peleenet.pth.tar')
















