import torch.nn as nn
from torch.hub import load_state_dict_from_url


import sys
sys.path.append(".") 
from .se_module import SELayer
from ..MobileNet.mobilenetv2 import MobileNetV2

class SEBasicBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, reduction = 16):
        super(SEBasicBlock, self).__init__()
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
        self.se = SELayer(oup, reduction)

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = self.se(x)

        if self.use_res_connect:
            return x + input
        else:
            return x


def se_mobilenetv2( model_name, class_num):
    model = MobileNetV2(n_class = class_num, InvertedResidual = SEBasicBlock).cuda()

    return model


if __name__ == '__main__':
    net = mobilenetv2_senet('../mobilenetv2_1.0-f2a8633.pth.tar')
    """
    x = torch.rand(1, 3, 256, 96).cuda()
    net_trace = torch.jit.trace(net,x)

    out1 = net(x)
    out2 = net_trace(x)
    print(net_trace.code)
    print(out1)
    print(out2)
    """