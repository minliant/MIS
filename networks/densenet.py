import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet']

#------DenseBlock-------#
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate


    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  #dropout是用来防止过拟合的
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,kernel_size=1, stride=1, bias=False))  #1*1卷积改变通道数

        self.add_module('pool_norm', nn.BatchNorm2d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv2d(num_output_features, num_output_features, kernel_size=2, stride=2))  #改变特征图大小

'''
class _Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
'''

class DenseNet(nn.Module):
    r"""
    :
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0, num_classes=1):

        super(DenseNet, self).__init__()

        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),  #不改变特征图大小
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
        self.conv_pool_first = nn.Conv2d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                         bias=False)

        # Each denseblock
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])  #创建空结构，后期append进去
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])

        for i, num_layers in enumerate(block_config):  #block_config=(6, 12, 24, 16) ，enumerate返回下标（0开始）和内容
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block)  #加入denseblock

            num_features = num_features + num_layers * growth_rate
            #上采样
            up_block = nn.ConvTranspose2d(num_features, num_classes, kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, groups=num_classes, bias=False)

            self.upsampling_blocks.append(up_block)  #加入上采样结构

            if i != len(block_config) - 1:

                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                #self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.se = Squeeze_Excite(num_init_features, 10)


        # ----------------------- classifier -----------------------
        self.bn_class = nn.BatchNorm2d(num_classes * 4  +num_init_features)
        self.conv_class = nn.Conv2d(num_classes * 4+num_init_features , num_classes, kernel_size=1, padding=0)
        # ----------------------------------------------------------


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
                #nn.Conv3d.bias.data.fill_(-0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.se(self.features_bn(first_three_features))

        out = self.conv_pool_first(first_three_features_bn)

        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)

        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)

        out = self.dense_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)

        out = self.dense_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)
        #
        out =  torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)

        # ----------------------- classifier -----------------------
        out = self.conv_class(F.relu(self.bn_class(out)))
        # ----------------------------------------------------------

        return out

class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  #全局平均池化
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
