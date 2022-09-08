import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1,use_relu = True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        out = self.convs(x)
        return out

class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)

        self.stem_2a = Conv_bn_relu(num_init_features,int(num_init_features/2),1,1,0)

        self.stem_2b = Conv_bn_relu(int(num_init_features/2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features*2, num_init_features, 1,1,0)


    def forward(self, x):
        stem_1_out  = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))

        return out

class TwoDenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growthRate):
        super(TwoDenseBlock, self).__init__()

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

    def forward(self, x):
        left_feat = self.left(x)
        
        right_feat = self.right(x)

        cat_feat = torch.cat([x,left_feat,right_feat],1)
        
        return cat_feat

class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling= True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup,1,1,0),
                                    nn.AvgPool2d(kernel_size=2,stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1,1,0)

    def forward(self, x):
        
        out = self.tb(x)
        
        return out

class PeleeDenseStage(nn.Module):
    def __init__(self, inp, growthRate, nDenseBlock, BottleneckWidth, pool, TwoDenseBlock = TwoDenseBlock):
        super(PeleeDenseStage, self).__init__()

        self.half_growthRate = int(growthRate / 2)
        self.densestage = nn.Sequential()

        current_ch = inp
        inter_ch = int(growthRate // 2 * BottleneckWidth / 4) * 4
        for i in range(nDenseBlock):
            self.densestage.add_module(
                "dense{}".format(i + 1),
                TwoDenseBlock(current_ch, inter_ch, self.half_growthRate)
            )
            current_ch += growthRate
        
        self.densestage.add_module(
            "transition",
            TransitionBlock(current_ch, current_ch, pool)
        )

    def forward(self, x):
        
        x = self.densestage(x)
        
        return x

class CSPDenseStage(PeleeDenseStage):
    def __init__(self, inp, growthRate, nDenseBlock, BottleneckWidth, pool, partial_ratio, TwoDenseBlock = TwoDenseBlock):
        split_ch = int(inp * partial_ratio)

        super(CSPDenseStage, self).__init__(split_ch, growthRate, nDenseBlock, BottleneckWidth, pool, TwoDenseBlock)

        self.split_ch = split_ch
        current_ch = inp + (growthRate * nDenseBlock)
        self.transition2 = TransitionBlock(current_ch, current_ch, with_pooling = pool)

    def forward(self, x):
        x1 = x[:, :self.split_ch, ...]
        x2 = x[:, self.split_ch:, ...]
        
        feat1 = self.densestage(x1)
        
        feat = torch.cat([x2, feat1], dim=1)
        feat = self.transition2(feat)
        return feat

class PeleeNet(nn.Module):
    def __init__(self,num_classes=11, num_init_features=32, growthRate=32, nDenseBlocks = [3,4,8,6], bottleneck_width=[1,2,4,4], partial_ratio = 1.0, TwoDenseBlock = TwoDenseBlock):
        super(PeleeNet, self).__init__()

        #1.stem 
        self.stem = StemBlock()
        self.stages = nn.Sequential()
        self.num_classes = num_classes
        current_ch = num_init_features

        self.half_growth_rate = int(growthRate / 2)
        
        #
        for i, (nDenseBlock, BottleneckWidth) in enumerate(zip(nDenseBlocks, bottleneck_width)):

            if i == len(nDenseBlocks)-1:
                with_pooling = False
            else:
                with_pooling = True
            if partial_ratio < 1.0:
                with_pooling = False
                stage = CSPDenseStage(current_ch, growthRate, nDenseBlock, BottleneckWidth, with_pooling, partial_ratio, TwoDenseBlock = TwoDenseBlock)
            else:
                stage = PeleeDenseStage(current_ch, growthRate, nDenseBlock, BottleneckWidth, with_pooling, TwoDenseBlock = TwoDenseBlock)
            current_ch += growthRate * nDenseBlock
            self.stages.add_module('stage_{}'.format(i+1),stage)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(current_ch, self.num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        # global average pooling layer
        x = self.pool(x).reshape(x.shape[0], -1)
        
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.zeros_(m.bias)

def csppeleenet(model_name, class_num):
    model_dict = {
        'peleenet_csp':PeleeNet(num_classes = class_num, partial_ratio = 0.5),
        'peleenet':PeleeNet(num_classes = class_num, partial_ratio = 1.0)
    }
    model = model_dict[model_name].cuda()
    
    return model

def CAM_CSPFeatures(input, model_path):
    model = torch.load(model_path).cpu()
    #print(model)
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
















