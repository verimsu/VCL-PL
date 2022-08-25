from collections import namedtuple
import csv
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.tf_a = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(hue=.05, saturation=.05),
            ], p=0.8),
            transforms.RandomGrayscale(0.2),
        ])        
        if mode == 'train':
            self.images = images[:1627]
            self.labels = labels[:1627]

        if mode == 'valid':
            self.images = images[162770:182637]
            self.labels = labels[162770:182637]

        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.images[index]))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att.to(torch.float32)
    def __len__(self):
        return self.length(base)