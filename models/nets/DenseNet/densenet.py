import math
import torch
import torch.nn as nn
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

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

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

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), in_planes=64, drop_rate=0, num_classes=1000):
        """ 
        param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper        
        :param block_config: (list of 4 ints) number of layers in each DenseBlock       
        :param num_init_features: (int) number of filters in the first Conv2d        
        :param bn_size: (int) the factor using in the bottleneck layer        :param compression_rate: (float) the compression rate used in Transition Layer        
        :param drop_rate: (float) the drop rate after each DenseLayer        :param num_classes: (int) number of classes for classification       
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, in_planes, kernel_size=7, stride=2,
                               padding=3, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1)
                )
        
        # DenseBlock
        self.features = nn.Sequential()
        num_features = in_planes
        #block_config每个block的denselayer层数
        for i, num_layers in enumerate(block_config):
            
            block = DenseBlock(num_layers, num_features, growth_rate, BottleneckBlock, drop_rate)
            self.features.add_module("denseblock_{}".format(i+1), block)
            num_features += num_layers*growth_rate
            if i == len(block_config)-1:
                with_pooling = True
                transition = TransitionBlock(num_features, num_features, with_pooling= with_pooling)
                self.features.add_module("transition_{}".format(i+1), transition)
            
        # final bn+ReLU
        self.adapool = nn.Sequential(
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
        )

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.adapool(x).reshape(x.shape[0], -1)
        
        x = self.classifier(x)
        return x

    # params initialization
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

def densenet(model_name, class_num):
    model_dict = {
        'densenet_40':DenseNet( growth_rate=12, block_config=(6, 6, 6), num_classes=class_num),
        'densenet_121':DenseNet( growth_rate=32, block_config=(6, 12, 24, 16), num_classes=class_num),
        'densenet_169':DenseNet( growth_rate=32, block_config=(6, 12, 32, 32), num_classes=class_num),
        'densenet_201':DenseNet( growth_rate=32, block_config=(6, 12, 48, 32), num_classes=class_num),
        'densenet_161':DenseNet( growth_rate=48, block_config=(6, 12, 36, 24), num_classes=class_num),
    }
    model = model_dict[model_name].cuda()
    
    return model