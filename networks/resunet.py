import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['ResUNet']


class ResUNet(nn.Module):

    def __init__(self, num_init_features=32, num_classes=1,up_channels=8):

        super(ResUNet, self).__init__()
        nb_c = [64,128,256,512]

        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.features_bn = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm2d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv2d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,bias=False)

        self.resblock0 = ResBlock(num_init_features)
        self.tr0 = Transition(num_init_features,nb_c[0])
        self.resblock1 = ResBlock(nb_c[0])
        self.tr1 = Transition(nb_c[0],nb_c[1])
        self.resblock2 = ResBlock(nb_c[1])
        self.tr2 = Transition(nb_c[1],nb_c[2])
        self.resblock3 = ResBlock(nb_c[2])
        self.tr3 = Transition(nb_c[2],nb_c[3])

        self.up_block0 = nn.ConvTranspose2d(num_init_features,up_channels,kernel_size=4,stride=2,padding=1,groups=up_channels,bias=False)
        self.up_block1 = nn.ConvTranspose2d(nb_c[0],up_channels,kernel_size=6,stride=4,padding=1,groups=up_channels,bias=False)
        self.up_block2 = nn.ConvTranspose2d(nb_c[1],up_channels,kernel_size=10,stride=8,padding=1,groups=up_channels,bias=False)
        self.up_block3 = nn.ConvTranspose2d(nb_c[2],up_channels,kernel_size=18,stride=16,padding=1,groups=up_channels,bias=False)

        self.se = Squeeze_Excite(num_init_features,8)


        # ----------------------- classifier -----------------------#
        self.bn_class = nn.BatchNorm2d(up_channels * 4  +num_init_features)
        self.conv_class = nn.Conv2d(up_channels * 4+num_init_features , num_classes, kernel_size=1)
        # ----------------------------------------------------------#


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.se(self.features_bn(first_three_features))

        out = self.conv_pool_first(first_three_features_bn)  #特征图开始缩小

        out = self.resblock0(out)
        up_block1 = self.up_block0(out)
        out = self.tr0(out)

        out = self.resblock1(out)
        up_block2 = self.up_block1(out)
        out = self.tr1(out)

        out = self.resblock2(out)
        up_block3 = self.up_block2(out)
        out = self.tr2(out)

        out = self.resblock3(out)
        up_block4 = self.up_block3(out)

        out =  torch.cat([up_block4, up_block3, up_block2, up_block1, first_three_features], 1)

        # ----------------------- classifier -----------------------#
        out = self.conv_class(F.relu(self.bn_class(out)))
        # ----------------------------------------------------------#

        return out


class ResBlock(nn.Module):
    def __init__(self,input_channels):
        super(ResBlock,self).__init__()
        self.conv0 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn0 = nn.BatchNorm2d(input_channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.se = Squeeze_Excite(input_channels,8)


    def forward(self,x):
        x0 = x
        x1 = self.relu0(self.bn0(self.conv0(x0)))
        x2 = self.relu1(self.bn1(self.conv1(x1)))

        out0 = x2 + x0
        out1 = self.se(out0)

        return out1


class Transition(nn.Module):
    def __init__(self,num_input_features,num_output_features):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features,num_output_features,kernel_size=1,stride=1,bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)

        return out

class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction):
        super(Squeeze_Excite,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(channel // reduction, channel, bias=False),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
