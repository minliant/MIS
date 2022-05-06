import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import networks
import albumentations as A
import numpy as np
from collections import OrderedDict
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from glob import glob
from torch.nn import functional as F
from medpy.metric.binary import dc,hd95
from torch.utils.data import DataLoader
from dataloaders.adem_dataset import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import utils as vutils
from torch.optim import lr_scheduler
from utils.tools import str2bool,AverageMeter,save_checkpoint
from utils import losses,metrics
from matplotlib import pyplot as plt

'''指令
python train.py --mode train --arch SSPUNet --batch_size 4 --epoch 300  --lr 0.01 
'''
'''服务器IP
10.63.110.208
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''---------动态模型参数调整---------'''
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', type=int,default=10,metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument("--config", type=str, default='./configs')
parser.add_argument('--model_path', type=str,default='./configs/model_cur.pth')
parser.add_argument('--result_csv', default='./result.csv')
# model
parser.add_argument('--arch', type=str,
                    default='UNet', help='model architecture')
parser.add_argument('--input_channels',type=int,default=3,
                    help='input_channels')
parser.add_argument('--num_classes',type=int,default=1,
                    help='number of classification classes')
parser.add_argument('--input_w',type=int,default=512,
                    help='image width')
parser.add_argument('--input_h',type=int,default=512,
                    help='image heigt')
# loss
parser.add_argument('--loss',type=str,default='GeneralizedDiceLoss',
                    help='Lossfunction')
# optimizer
parser.add_argument('--optimizer',choices=['Adam','SGD'],default='SGD',
                    help='optimizer')
parser.add_argument('--lr',type=float,default=1e-3, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', type=float,default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay',type=float,default=1e-4,
                    help='weight decay')
# scheduler
parser.add_argument('--scheduler', default='CosineAnnealingLR',
                    choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
parser.add_argument('--min_lr', type=float,default=1e-5,
                    help='minimum learning rate')
parser.add_argument('--num_workers', type=int,default=4) #进程数
# datasets
parser.add_argument('--dataset',type=str,default='ADEM')

arg = parser.parse_args()


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter(),
                  # 'hausdorff': AverageMeter()
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input,target in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        # output = output[0]
        loss = criterion(output,target).to(device)
        # loss2 = F.binary_cross_entropy_with_logits(output,target).to(device)
        # loss = 0.5 * loss1 + 0.5 * loss2

        # dice = dc(output.cpu().detach().numpy(), target.cpu().detach().numpy())
        dice = metrics.dice(output,target)
        # hausdorff = hd95(output.cpu().detach().numpy(), target.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        avg_meters['loss'].update(loss.item())
        avg_meters['dice'].update(dice)
        # avg_meters['hausdorff'].update(hausdorff, input.size(0))

        postfix = OrderedDict([
            ('loss', loss.item()),
            ('dice', dice),
            # ('hausdorff', avg_meters['hausdorff'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg),
                        # ('hausdorff', avg_meters['hausdorff'].avg)
                        ])

def evaluate(val_loader,model,criterion):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter(),
                  # 'hausdorff': AverageMeter()
                  }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            # output = output[0]
            loss = criterion(output, target).to(device)
            # loss2 = F.binary_cross_entropy_with_logits(output, target).to(device)
            # loss = 0.5 * loss1 + 0.5 * loss2

            # dice = dc(output.cpu().detach().numpy(), target.cpu().detach().numpy())
            dice = metrics.dice(output, target)
            # hausdorff = hd95(output.cpu().detach().numpy(), target.cpu().detach().numpy())

            vutils.save_image(output[0], 'output.jpg', normalize=False)
            vutils.save_image(input[0], 'input.jpg', normalize=True)
            vutils.save_image(target[0], 'target.jpg', normalize=False)

            avg_meters['loss'].update(loss.item())
            avg_meters['dice'].update(dice)
            # avg_meters['hausdorff'].update(hausdorff, input.size(0))

            postfix = OrderedDict([
                ('loss', loss.item()),
                ('dice', dice),
                # ('hausdorff', avg_meters['hausdorff'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg),
                        # ('hausdorff', avg_meters['hausdorff'].avg)
                        ])

def main():
    args = arg

    cudnn.benchmark = True  # 增加模型训练速度
    os.makedirs('configs/models/%s' % args.arch, exist_ok=True)
    tagger = 0
    if os.path.exists('%s.txt' % args.arch):
        os.remove('%s.txt' % args.arch)

    print("=> creating model %s" % args.arch)
    model = networks.__dict__[args.arch]()
    model = model.to(device)

    '''---------设置损失函数---------'''
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)

    '''---------设置优化器----------'''
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    '''---------设置调度器---------'''
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr)
    else:
        raise NotImplementedError

    '''---------数据载入---------'''
    img_ids = glob(os.path.join('data', args.dataset, 'images', '*' + '.nrrd'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    #数据增强统一数据尺寸
    train_transform = Compose([
        transforms.Flip(),
        OneOf([
            A.augmentations.transforms.HueSaturationValue(),
            A.augmentations.transforms.RandomBrightness(),
            A.augmentations.transforms.RandomContrast(),
        ], p=1),
        A.augmentations.Resize(args.input_h, args.input_w),
        A.augmentations.transforms.Normalize(),
    ])

    val_transform = Compose([
        A.augmentations.Resize(args.input_h, args.input_w),
        A.augmentations.transforms.Normalize(),
    ])

    train_dataset = Dataset(train_img_ids,
                            img_dir=os.path.join('data',args.dataset,'images'),
                            mask_dir=os.path.join('data',args.dataset,'masks'),
                            num_classes=args.num_classes,
                            transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)

    val_dataset = Dataset(val_img_ids,
                          img_dir=os.path.join('data',args.dataset,'images'),
                          mask_dir=os.path.join('data',args.dataset,'masks'),
                          num_classes=args.num_classes,
                          transform=val_transform)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            drop_last=True)

    epo = []
    train_dice_list = []  #用于生成可视化图像
    val_dice_list = []

    '''---------训练模型---------'''
    for epoch in range(args.epoch):
        print('Epoch [%d/%d]' % (epoch, args.epoch))
        epo.append(epoch)

        train_log = train(train_loader, model, criterion,optimizer)
        val_log = evaluate(val_loader, model, criterion)

        train_dice_list.append(train_log['dice'])
        val_dice_list.append(val_log['dice'])

        with open('%s.txt' % args.arch,'a',encoding='utf-8') as f:  #with open 会自动调用close
            f.write('%d \t %f \t %f \n' % (epoch,train_log['dice'],val_log['dice']))

        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()

        print(
            'loss %.4f  -- dice %.4f -- val_loss %.4f   -- val_dice %.4f '
            % (train_log['loss'],
               train_log['dice'],
               # train_log['hausdorff'],
               val_log['loss'],
               val_log['dice'],
               # val_log['hausdorff']
               ))

        '''---------保存最优模型参数---------'''
        if (val_log['dice'] >= tagger):
            tagger = val_log['dice']
            torch.save(model.state_dict(), 'configs/models/%s/model.pth' % args.arch)
            best_loss = val_log['loss']
            print("=> saved best model")

        '''---------每50个epoch保存模型及超参数---------'''
        if (epoch == 49):
            save_checkpoint({
                'fold': 0,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss':best_loss
            }, single=True, checkpoint=args.config)

        torch.cuda.empty_cache()

    plt.plot(epo, train_dice_list)
    plt.plot(epo, val_dice_list,color='red')
    plt.savefig('Dice_curve.png', c='c', transparent=True)


if __name__ == '__main__':
    main()
