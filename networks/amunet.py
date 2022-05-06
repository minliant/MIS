from collections import OrderedDict
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init

__all__ = ['AMUNet']

class AMUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super(AMUNet,self).__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024, 2048]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = nn.Conv2d(in_channels=nb_filter[0],out_channels=nb_filter[1],kernel_size=3,padding=1)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=nb_filter[1], out_channels=nb_filter[1], kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(in_channels=nb_filter[2], out_channels=nb_filter[1], kernel_size=3, padding=1)
        # self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels=nb_filter[3], out_channels=nb_filter[1], kernel_size=3, padding=1)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        # self.cam = CAM() # 64*4 = 256
        # self.sem = SEM()
        self.aspp = ASPP(nb_filter[3],nb_filter[0])
        self.conv = nn.Conv2d(nb_filter[2],nb_filter[0],kernel_size=3,padding=1)
        # self.ssm = SSM()
        # self.merge = Merge()

        self.final = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
        # self.final_f = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input) # 512*512*32

        x1_0 = self.conv1_0(self.pool(x0_0)) # 256*256*64
        x2_0 = self.conv2_0(self.pool(x1_0)) # 128*128*128
        x3_0 = self.conv3_0(self.pool(x2_0)) # 64*64*256
        x4_0 = self.conv4_0(self.pool(x3_0)) # 32*32*512
        x04 = self.conv3(self.up2(x3_0)) # 512*512*64
        x03 = self.conv2(self.up1(x2_0)) # 512*512*64
        x02 = self.conv1(self.up0(x1_0)) # 512*512*64
        x01 = self.conv0(x0_0) # 512*512*64
        c2_5 = torch.cat([x01,x02,x03,x04],1) # 512*512*64*4 = 512*512*256
        asp = self.aspp(c2_5) # 512*512*128
        out1 = self.conv(asp)
        # cam = self.cam(c2_5) #cam是一个list，cam[0] = 512*512*1，cam[1] = 256*256*1，cam[2] = 128*128*1，cam[3] = 64*64*1


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        out = torch.cat([out1,x0_4],1)

        output = self.final(out)
        # output = self.final_f(out)

        return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1 # kernel_size = (1,3,3,1)
            padding = dilation if dilation > 1 else 0 # padding = (0,1,1,0)   dilations=(1, 2, 5, 1)
            conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size ,dilation=dilation,padding=padding,bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.bn(self.aspp[aspp_idx](inp))))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out

class CAM(nn.Module):
    def __init__(self, inplanes=256, output_channels=64):
        super(CAM, self).__init__()
        self.dila_conv = nn.Sequential(nn.Conv2d(inplanes, output_channels,kernel_size=3, stride=1, padding=1),
                                       ASPP(output_channels, output_channels // 4),
                                       nn.Conv2d(output_channels, output_channels,kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(output_channels),
                                       nn.ReLU(inplace=False)
                                       )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.Conv2d(output_channels,1,kernel_size=3,stride=1,padding=1)
        down_stride = (1,2,2,2)
        for i in down_stride:
            self.down_conv.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=i,padding=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):

        lvl_fea = self.dila_conv(x)
        multi_atts = []
        for i in range(4):   #生成四个heatmap
            lvl_fea = self.down_conv[i](lvl_fea)
            lvl_att = self.att_conv(lvl_fea)
            multi_atts.append(self.sigmoid(lvl_att))
        return multi_atts

class SEM(nn.Module):
    def __init__(self):
        super(SEM,self).__init__()

    def forward(self,x,heatmap):
        out = x * heatmap
        out = out + x
        return out
