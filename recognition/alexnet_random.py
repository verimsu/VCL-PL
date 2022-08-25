#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
from torchsummary import summary
import config
from facenet_pytorch import training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import glob
from utils.collate import collate_custom
import torchvision.models as models
from util import AverageMeter, learning_rate_decay, Logger
import collections

# In[ ]:

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# Root directory for dataset
data_root = "/home/mehmetyavuz/datasets/CelebA128/"
attr_root = "/home/mehmetyavuz/datasets/list_attr_celeba.txt"
# Number of workers for dataloader
workers = 8

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = (128,128)
epochs = 100


# In[ ]:


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
            self.images = images[:16277]
            self.labels = labels[:16277]

        if mode == 'valid':
            self.images = images[162770:182637]
            self.labels = labels[162770:182637]

        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        if index < 16277:
            img = self.tf(self.tf_a(Image.open(os.path.join(self.data_path, self.images[index]))))
        else:
            img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att.to(torch.float32)
    def __len__(self):
        return self.length


# In[ ]:


attrs_default = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


# In[ ]:


dataset = CelebA(data_root, attr_root, image_size, 'train', attrs_default)
train_loader = torch.utils.data.DataLoader(dataset, num_workers=workers, 
                batch_size=batch_size, pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)
dataset = CelebA(data_root, attr_root, image_size, 'valid', attrs_default)
val_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=workers)
dataset = CelebA(data_root, attr_root, image_size, 'test', attrs_default)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=workers)


# In[ ]:


# Decide which device we want to run on
device = torch.device("cuda:0")


# In[ ]:

resnet = models.__dict__['alexnet'](pretrained=False)
resnet.classifier[6] = nn.Linear(4096,40,bias=True)
resnet = torch.nn.DataParallel(resnet)
resnet.cuda()

# In[ ]:


optimizer = optim.Adam(resnet.parameters(), lr=0.00001)
scheduler = None


# In[ ]:


loss_fn = torch.nn.BCEWithLogitsLoss()
metrics = {
    'acc': training.accuracy_ml
} 


# In[ ]:


print('\n\nInitial')
print('-' * 10)

val_loss = 1
for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()        
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        #writer=writer
    )
    
    #if epoch + 1 >= 30:
    resnet.eval()
    val_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            #writer=writer
        )

    if val_metrics[0].item() < val_loss:
        val_loss = val_metrics[0].item()
        print('Test set Accuracy Lowest Validation Loss:')
        training.pass_epoch(
                resnet, loss_fn, test_loader,
                batch_metrics=metrics, show_running=True, device=device,
                #writer=writer
            )
        torch.save(resnet.state_dict(), "alexnet_random_010_0.pth")

#writer.close()


# In[ ]:





# In[ ]:





# In[ ]:




