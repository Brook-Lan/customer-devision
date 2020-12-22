#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-15

@author:LHQ
"""
import os
import random

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T


def make_transform(resize=128, normal_mean=[0.485, 0.456,0.406], normal_std=[0.229,0.224,0.225]):
    transform = T.Compose([
                    T.Resize((resize, resize)),
                    T.ToTensor(),
                    T.Normalize(mean=normal_mean, std=normal_std)
                ])
    return transform


class ContourDataset(Dataset):
    def __init__(self,
            root,
            resize=128,
            normal_mean=[0.485, 0.456,0.406],
            normal_std=[0.229,0.224,0.225],
            test=False,
            test_size=0.3
        ):
        random.seed(8)
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        random.shuffle(imgs)
        test_count = int(len(imgs) * test_size)
        if test:
            self.imgs = imgs[-test_count:]
        else:
            self.imgs = imgs[:test_count]

        self.transform = make_transform(resize, normal_mean, normal_std)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        filename = os.path.split(img_path)[-1]
        img_name = os.path.splitext(filename)[0]
        data = Image.open(img_path)
        data = self.transform(data)
        return data, img_name

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = ContourDataset("data/img/")
    img = dataset[0]
    print(img.shape)


