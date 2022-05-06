from collections import OrderedDict
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


__all__ = ['UNet']

class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024, 2048]

        self.pool = nn.MaxPool2d(2, 2)
        # self.sa = SpatialGgate()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)



    def forward(self, input):
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,drop_rate=0.5):
        super(VGGBlock,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.drop = Drop(drop_prob=drop_rate)
        # self.sa = SpatialGgate()
        # self.ca = ChannelGate(out_channels)
        # self.cbam = CBAM(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.drop(out)
        # out = self.sa(out)
        # out = self.ca(out)
        # out = self.cbam(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.drop(out)
        # out = self.sa(out)
        # out = self.ca(out)
        # out = self.cbam(out)

        return out

'''-----channel attention block-----'''
class ChannelGate(nn.Module):
    def __init__(self,gate_channel,ratio=8):
        super(ChannelGate,self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(gate_channel,gate_channel // ratio),
                                nn.ReLU(inplace=True),
                                nn.Linear(gate_channel // ratio,gate_channel))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b, c, h, w = x.size()
        out1 = self.fc(self.pool1(x).view(b,c))
        out2 = self.fc(self.pool2(x).view(b,c))
        out = out1 + out2
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

'''-----spatial attention block-----'''
class SpatialGgate(nn.Module):
    def __init__(self,kernel=7):
        super(SpatialGgate,self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=kernel,padding=(kernel - 1) // 2,bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x,1).unsqueeze(1)
        max_out = torch.max(x,1)[0].unsqueeze(1)
        out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    def __init__(self,input_channel):
        super(CBAM,self).__init__()
        self.ca = ChannelGate(input_channel)
        self.sa = SpatialGgate()

    def forward(self,x):
        out = self.ca(x)
        out = self.sa(out)
        return out




class Drop(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(Drop, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if self.drop_prob == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0,对应第五步
        # 这样计算可以得到丢弃的比率的随机点个数
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask