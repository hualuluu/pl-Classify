import torch.nn as nn
import torch
import math, copy


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=11, input_size=256, width_mult=1.):
        super(MobileNetV2, self).__init__()
        #block = torch.jit.script(InvertedResidual(), x）
        block = InvertedResidual
        se_block = SELayer
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features0 = [conv_bn(3, input_channel, 2)]
        self.features0 = nn.Sequential(*self.features0)
        #print(interverted_residual_setting[0], input_channel, width_mult)
        self.feature1, input_channel = self.make_feature(interverted_residual_setting[0], block, input_channel, width_mult)
        self.feature1 = nn.Sequential(*self.feature1)

        self.feature2, input_channel = self.make_feature(interverted_residual_setting[1], block, input_channel, width_mult)
        self.feature2 = nn.Sequential(*self.feature2)

        self.feature3, input_channel = self.make_feature(interverted_residual_setting[2], block, input_channel, width_mult)
        self.feature3 = nn.Sequential(*self.feature3)
        self.se3 = se_block(input_channel)

        self.feature4, input_channel = self.make_feature(interverted_residual_setting[3], block, input_channel, width_mult)
        self.feature4 = nn.Sequential(*self.feature4)
        self.se4 = se_block(input_channel)

        self.feature5, input_channel = self.make_feature(interverted_residual_setting[4], block, input_channel, width_mult)
        self.feature5 = nn.Sequential(*self.feature5)
        self.se5 = se_block(input_channel)

        self.feature6, input_channel = self.make_feature(interverted_residual_setting[5], block, input_channel, width_mult)
        self.feature6 = nn.Sequential(*self.feature6)
        self.se6 = se_block(input_channel)

        self.feature7, input_channel = self.make_feature(interverted_residual_setting[6], block, input_channel, width_mult)
        self.feature7 = nn.Sequential(*self.feature7)
        self.se7 = se_block(input_channel)
        """
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        """
        # building last several layers
        #self.features8 = [conv_1x1_bn(input_channel, self.last_channel)]
        self.features8 = [conv_1x1_bn(640, self.last_channel)]
        # make it nn.Sequential
        self.features8 = nn.Sequential(*self.features8)

        # building classifier
        self.classifier1 = nn.Linear(self.last_channel, n_class)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self._initialize_weights()
    def make_feature(self, interverted_residual_setting_ind, block, input_channel, width_mult):
        t, c, n, s = interverted_residual_setting_ind
        output_channel = make_divisible(c * width_mult) if t > 1 else c
        features = []
        for i in range(n):
            if i == 0:
                features.append(block(input_channel, output_channel, s, expand_ratio=t))
            else:
                features.append(block(input_channel, output_channel, 1, expand_ratio=t))
            input_channel = output_channel
        return copy.deepcopy(features), input_channel


    def forward(self, x):
        x = self.features0(x)

        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.se3(x)
        f1 = self.pool(x)

        x = self.feature4(x)
        x = self.se4(x)
        f1 = self.pool(x)

        x = self.feature5(x)
        x = self.se5(x)
        f2 = self.pool(x)

        x = self.feature6(x)
        x = self.se6(x)
        f3 = self.pool(x)

        x = self.feature7(x)
        x = self.se7(x)
        f4 = self.pool(x)

        f = torch.cat((f1,f2,f3,f4),1)
        
        x = self.features8(f)
        x = self.pool(x).reshape(x.shape[0], -1)

        x = self.classifier1(x)
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

temp_dict = {
    'features.1':'feature1.0',
    'features.2':'feature2.0',
    'features.3':'feature2.1',
    'features.4':'feature3.0',
    'features.5':'feature3.1',
    'features.6':'feature3.2',
    'features.7':'feature4.0',
    'features.8':'feature4.1',
    'features.9':'feature4.2',
    'features.10':'feature4.3',
    'features.11':'feature5.0',
    'features.12':'feature5.1',
    'features.13':'feature5.2',
    'features.14':'feature6.0',
    'features.15':'feature6.1',
    'features.16':'feature6.2',
    'features.17':'feature7.0'
}
def mobilenetv2_senet_mutifeature(pre_path, pretrained=True):
    model = MobileNetV2(width_mult=1).cuda()
    #
    
    if pre_path =='../mobilenetv2_1.0-f2a8633.pth.tar' or pre_path == '../output/mobilenet_0512/mobilenetv2_lr_state_dict29.pth':
        model_dict = model.state_dict()
        checkpoint = torch.load(pre_path)
        pretrained_dict = checkpoint
        state_dict = {}
        # 1. filter out unnecessary keys
        for k, v in pretrained_dict.items():
            #print(k)
            if k.split('.')[1] == '0':
                state_dict['features0.0.0.weight'] = pretrained_dict['features.0.0.weight']
                state_dict['features0.0.1.weight'] = pretrained_dict['features.0.1.weight']
                state_dict['features0.0.1.bias'] = pretrained_dict['features.0.1.bias']
                state_dict['features0.0.1.running_mean'] = pretrained_dict['features.0.1.running_mean']
                state_dict['features0.0.1.running_var'] = pretrained_dict['features.0.1.running_var']
                continue
            """
            if k.split('.')[1] == '18':
                state_dict['features8.0.0.weight'] = pretrained_dict['features.18.0.weight']
                state_dict['features8.0.1.weight'] = pretrained_dict['features.18.1.weight']
                state_dict['features8.0.1.bias'] = pretrained_dict['features.18.1.bias']
                state_dict['features8.0.1.running_mean'] = pretrained_dict['features.18.1.running_mean']
                state_dict['features8.0.1.running_var'] = pretrained_dict['features.18.1.running_var']
                continue
            """
            if len(k.split('.conv')) > 1:
                #print(temp_dict[k.split('.conv')[0]] + '.conv' + k.split('.conv')[1])
                state_dict[temp_dict[k.split('.conv')[0]] + '.conv' + k.split('.conv')[1]] = pretrained_dict[k]
                
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        """"""
    else:
        model = torch.load(pre_path)
    return model


if __name__ == '__main__':
    net = mobilenetv2_senet_mutifeature('../mobilenetv2_1.0-f2a8633.pth.tar')
    """
    x = torch.rand(1, 3, 256, 96).cuda()
    net_trace = torch.jit.trace(net,x)

    out1 = net(x)
    out2 = net_trace(x)
    print(net_trace.code)
    print(out1)
    print(out2)
    """