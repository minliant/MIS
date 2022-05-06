import numpy as np
import torch
from medpy import metric

# __all__ = ['dice']

def dice(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()  #view(-1)就是把数据展成一维
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
