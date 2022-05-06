import argparse
import os
import torch

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

#保存模型参数
def save_checkpoint(state, single=True, checkpoint='checkpoint'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    model_name = 'model_' + str(state['epoch']) + '.pth'
    model_path = os.path.join(checkpoint, fold + model_name)
    torch.save(state['state_dict'], model_path)


