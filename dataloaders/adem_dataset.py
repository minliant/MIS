import os
import nrrd
import numpy as np
import torch.utils.data
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, num_classes,  img_ext=".nrrd", mask_ext=".seg.nrrd", transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.文件后缀名
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        image_data, image_options = nrrd.read(os.path.join(self.img_dir, img_id + self.img_ext))
        label_data, label_options = nrrd.read(os.path.join(self.mask_dir, img_id + self.mask_ext))

        label_data = label_data[:, :, np.newaxis]

        if self.transform is not None:
            augmented = self.transform(image=image_data, mask=label_data)
            img = augmented['image']
            # img = img[80:432, 80:432]
            # img = img.crop((70,100,425,415))
            mask = augmented['mask']
            # mask = mask[80:432, 80:432]
            # mask = mask.crop((70,100,425,415))

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)

        return img, mask
