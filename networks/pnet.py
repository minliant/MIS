import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class PNet(nn.Module):
    def __init__(self,num_classes=1):
        super(PNet,self).__init__()
        nb = [32,64,128,256,512,1024]
        dilations = [1,2,5]
        paddings = [1,2,5]
        self.first_block = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3,nb[0],kernel_size=3,padding=paddings[2],dilation=dilations[2])),
            ('norm0',nn.BatchNorm2d(nb[0])),
            ('relu0',nn.ReLU(inplace=True)),
            ('conv1',nn.Conv2d(nb[0],nb[1],kernel_size=3,padding=paddings[2],dilation=dilations[2])),
            ('norm1',nn.BatchNorm2d(nb[1])),
            ('relu1',nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(nb[1], nb[2], kernel_size=3, padding=paddings[2],dilation=dilations[2])),
            ('norm2', nn.BatchNorm2d(nb[2])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(nb[2], nb[3], kernel_size=3, padding=paddings[2],dilation=dilations[2])),
            ('norm3', nn.BatchNorm2d(nb[3])),
            ('relu3', nn.ReLU(inplace=True)),
        ]))

        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv = nn.Conv2d(nb[3],nb[1],kernel_size=1)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.last_block = nn.Sequential(OrderedDict([
            ('c0', nn.Conv2d(nb[4],nb[3],kernel_size=1)),
            ('n0', nn.BatchNorm2d(nb[3])),
            ('r0', nn.ReLU(inplace=True)),
            ('c1', nn.Conv2d(nb[3],nb[2],kernel_size=1)),
            ('n1', nn.BatchNorm2d(nb[2])),
            ('r1', nn.ReLU(inplace=True)),
            ('c2', nn.Conv2d(nb[2],nb[1],kernel_size=1)),
            ('n2', nn.BatchNorm2d(nb[1])),
            ('r2', nn.ReLU(inplace=True)),
            ('c3', nn.Conv2d(nb[1],num_classes,kernel_size=1)),
        ]))

    def forward(self,x): # x的size为512*512*3
        first_conv = self.first_block(x) # 512*512*256
        x10 = self.pool(first_conv) # 256*256*256
        x20 = self.pool(x10) # 128*128*256
        x30 = self.pool(x20) # 64*64*256
        x40 = self.pool(x30) # 32*32*256

        x11 = self.conv(x10) # 256*256*64
        x21 = self.conv(x20) # 128*128*64
        x31 = self.conv(x30) # 64*64*64
        x41 = self.conv(x40) # 32*32*64

        x12 = self.up0(x11) # 512*512*64
        x22 = self.up1(x21) # 512*512*64
        x32 = self.up2(x31) # 512*512*64
        x42 = self.up3(x41) # 512*512*64

        out = torch.cat([first_conv,x12,x22,x32,x42],1) # 512*512*512

        output = self.last_block(out)

        return output


