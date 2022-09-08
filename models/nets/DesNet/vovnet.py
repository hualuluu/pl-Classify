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

class StemBlock(nn.Module):
    def __init__(self, inp = 3):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, 64, 3, 2)
        
        self.stem_2 = Conv_bn_relu(64, 64, 3, 1)

        self.stem_3 = Conv_bn_relu(64, 128, 3, 1)

        self.pool = PoolBlock(2)
    def forward(self, x):
        x  = self.stem_1(x)
        
        x = self.stem_2(x)
        
        x = self.stem_3(x)
        
        x = self.pool(x)

        return x

class PoolBlock(nn.Module):
    def __init__(self, _stride):
        super(PoolBlock, self).__init__()
        self.stride = _stride
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = self.stride, ceil_mode=True)

    def forward(self, x):
        
        x = self.maxpool(x)
        
        return x


class OSABlock(nn.Module):
    def __init__(self,
                 inp,
                 block_channel,
                 concat_channel,
                 nLayer = 5,
                 identity=False):
        super(OSABlock, self).__init__()

        self.identity = identity
        self.osablocks = nn.ModuleList()

        in_channel = inp
        for i in range(nLayer):
            osablock = Conv_bn_relu(in_channel, block_channel, 3, 1)
            in_channel = block_channel
            self.osablocks.add_module('osablock{}'.format(i+1), osablock)
        # feature aggregation
        in_channel = inp + nLayer * block_channel
        self.concat = Conv_bn_relu(in_channel, concat_channel, 1, 1, 0)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        #print(x.shape)
        for osablock in self.osablocks:
            #print(osablock)
            x = osablock(x)
            #print(x.shape)
            output.append(x)

        x = torch.cat(output, dim=1)
        x = self.concat(x)

        if self.identity:
            x = x + identity_feat

        return x


class OSAStage(nn.Module):
    def __init__(self,
                 inp,
                 block_channel,
                 concat_channel,
                 nOSABlock,
                 pStride,
                 nLayer = 5
                 ):
        super(OSAStage, self).__init__()

        self.osastage = nn.Sequential()
        in_channel = inp
        for i in range(nOSABlock):

            osastage = OSABlock(in_channel, block_channel, concat_channel, nLayer)
            self.osastage.add_module(
                 "osastage{}".format(i + 1),
                 osastage
                 )
        
        self.pool = PoolBlock(pStride)

    def forward(self, x):
        x = self.osastage(x)
        #print(x.shape)
        x = self.pool(x)

        return x

class VoVNet(nn.Module):
    def __init__(self,
                 num_classes,
                 nBlockChannels,
                 nConcatChannels,
                 nStageBlocks,
                 nPoolStrides = [2, 2, 2, 2],
                 nLayer = 5):
        super(VoVNet, self).__init__()

        # Stem module
        self.stem = StemBlock()
        self.num_classes = num_classes

        in_channel = 128
        self.osa = nn.Sequential()
        for i, (block_channel,concat_channel, nOSABlock, pStride) in enumerate(zip(nBlockChannels, nConcatChannels, nStageBlocks, nPoolStrides)):
            osa = OSAStage(in_channel,
                 block_channel,
                 concat_channel,
                 nOSABlock,
                 pStride,
                 nLayer
                 )
            in_channel = concat_channel
            self.osa.add_module('osa_{}'.format(i + 1), osa)
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channel, self.num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        #print(x.shape)
        x = self.osa(x)
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
                nn.init.zeros_(m.bias)


def vovnet(model_name, class_num):
    model_dict = {
        'vovnet_27_slim':VoVNet(num_classes = class_num,
                 nBlockChannels = [64, 80, 96, 112],
                 nConcatChannels = [128, 256, 384, 512],
                 nStageBlocks = [1,1,1,1]
                 ),
        'vovnet_39':VoVNet(num_classes = class_num,
                 nBlockChannels = [128, 160, 192, 224],
                 nConcatChannels = [256, 512, 768, 1024],
                 nStageBlocks = [1,1,2,2]
                 ),
        'vovnet_57':VoVNet(num_classes = class_num,
                 nBlockChannels = [128, 160, 192, 224],
                 nConcatChannels = [256, 512, 768, 1024],
                 nStageBlocks = [1,1,4,3]
                 )
    } 
    model = model_dict[model_name].cuda()

    return model

def CAM_VOVFeatures(input, model_path):
    model = torch.load(model_path).cpu()
    
    model.eval()
    x = model.stem(input)
    feature = model.osa(x)
    # global average pooling layer
    pool_feature = model.pool(feature)
        
    fc = model.classifier(pool_feature.reshape(pool_feature.shape[0], -1))
    return feature, fc