from collections import OrderedDict
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class DAN(nn.Module):
    def __init__(self,num_classes,input_channels):
        super(DAN, self).__init__()
        self.sn = SN(num_classes)
        self.en = EN(num_classes)

    def forward(self,x):
        out0 = self.sn(x)
        out1 = torch.cat([x,out0],1)
        out2 = self.en(out1)

        return out2


class SN(nn.Module):
    def __init__(self, num_classes):
        super(SN,self).__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024, 2048]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock1(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock1(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock1(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock1(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock1(nb_filter[3], nb_filter[4], nb_filter[4])


        self.conv3_1 = VGGBlock1(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock1(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock1(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock1(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

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


class EN(nn.Module):
    def __init__(self,num_classes=1):
        super(EN,self).__init__()
        n_ch = [64,128,256,512,1024]

        self.pool = nn.MaxPool2d(2,stride=2)
        self.bn = nn.BatchNorm2d(n_ch[4])
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = VGGBlock2(4,n_ch[0])
        self.conv1 = VGGBlock2(n_ch[0],n_ch[0])
        self.conv2 = VGGBlock2(n_ch[0],n_ch[1])
        self.conv3 = VGGBlock2(n_ch[1],n_ch[2])
        self.conv4 = VGGBlock2(n_ch[2],n_ch[2])
        self.conv5 = VGGBlock2(n_ch[2],n_ch[3])
        self.conv6 = VGGBlock2(n_ch[3],n_ch[3])
        self.conv7 = VGGBlock2(n_ch[3],n_ch[3])
        self.conv8 = VGGBlock2(n_ch[3],n_ch[3])
        self.L0 = nn.Linear(n_ch[3],n_ch[4])
        self.L1 = nn.Linear(n_ch[4],n_ch[4])
        self.L2 = nn.Linear(n_ch[4],n_ch[4])
        self.L3 = nn.Linear(n_ch[4],num_classes)


    def forward(self,x):
        x0 = self.conv0(x)
        x1 = self.pool(self.conv1(x0))
        x2 = self.pool(self.conv2(x1))
        x3 = self.conv4(self.conv3(x2))
        x4 = self.conv5(self.pool(x3))
        x5 = self.pool(self.conv6(x4))
        x6 = self.conv8(self.conv7(x5))

        x7 = self.L0(x6)
        x8 = self.L1(self.relu(self.bn(x7)))
        x9 = self.L2(self.relu(self.bn(x8)))
        x10 = self.relu(self.bn(x9))

        out = self.L3(x10)
        out = torch.sigmoid(out)

        return out


class VGGBlock1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock1,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.SE = Squeeze_Excite(out_channels, 8)  #这个8有待研究

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.SE(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.SE(out)

        return out

class VGGBlock2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(VGGBlock2,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
