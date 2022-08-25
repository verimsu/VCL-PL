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
import torchvision.models as models
from util import AverageMeter, learning_rate_decay, Logger
from utils.config import create_config
from utils.collate import collate_custom
from torchsummary import summary

# In[ ]:


p = create_config("configs/env.yml", "configs/pretext/pretraining.yml", 128, 100)
p['batch_size'] = 64
p['epochs'] = 100

# In[ ]:

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['waugmentation_kwargs']['color_jitter'])
            ], p=p['waugmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['waugmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['waugmentation_kwargs']['normalize'])
        ])


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
        if index < 162770:
            img = self.tf(self.tf_a(Image.open(os.path.join(self.data_path, self.images[index]))))
        else:
            img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att.to(torch.float32)
    def __len__(self):
        return self.length


# In[ ]:

# Number of workers for dataloader

workers = 8

# Batch size during training


image_size = (128,128)


# In[ ]:


from models.alexnet import alexnet
alexnet = alexnet()['backbone']


# In[ ]:


alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=40, bias=True)
alexnet = alexnet.cuda()
summary(alexnet,(3,128,128))

# In[ ]:


data_root = "/home/mehmetyavuz/datasets/CelebA128/"
attr_root = "/home/mehmetyavuz/datasets/list_attr_celeba.txt"


# In[ ]:


attrs_default = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", 
                 "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", 
                 "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", 
                 "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", 
                 "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

# In[ ]:


#dataset_one = CelebA(data_root, attr_root, image_size, 'train', attrs_default)
dataset_two = torchvision.datasets.ImageFolder(root='/home/mehmetyavuz/datasets/YFCC392K/', transform=transform)
y_YFCC = np.load('y_YFCC_weighted.npy')
#dataset_two.targets = [y_YFCC[i,:] for i in range(y_YFCC.shape[0])]
#train_dataset = torch.utils.data.ConcatDataset([dataset_one, dataset_two])
#
train_loader = torch.utils.data.DataLoader(dataset_two, num_workers=p['num_workers'], 
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)

dataset = CelebA(data_root, attr_root, image_size, 'valid', attrs_default)
val_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=p['batch_size'],
                                          shuffle=False,
                                          num_workers=workers)
dataset = CelebA(data_root, attr_root, image_size, 'test', attrs_default)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=p['batch_size'],
                                          shuffle=False,
                                          num_workers=workers)


# In[ ]:


# Decide which device we want to run on
device = torch.device("cuda:0")


# In[ ]:


alexnet = torch.nn.DataParallel(alexnet)


# In[ ]:


#weights = torch.load("results/CelebA/SimCLR-B128/finetuning_model.pth.tar")["model"]
#for key in list(weights.keys()):
#    weights[key.replace('module.backbone.', 'module.')] = weights.pop(key)
#    if 'cluster_head' in key:
#        del weights[key]


# In[ ]:


#alexnet.load_state_dict(weights,strict=False)


# In[ ]:


optimizer = optim.Adam(alexnet.parameters(), lr=0.0001)
#Q = math.floor(len(train_dataset)*p['epochs'])
scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Q)


# In[ ]:


loss_fn = torch.nn.BCEWithLogitsLoss()
metrics = {
    'acc': training.accuracy_ml
} 


# In[ ]:


print('\n\nInitial')
print('-' * 10)
alexnet.eval()
training.pass_epoch(
    alexnet, loss_fn, test_loader,
    batch_metrics=metrics, show_running=True, device=device,
    #writer=writer
)

val_loss = 1
for epoch in range(p['epochs']):
    print('\nEpoch {}/{}'.format(epoch + 1, p['epochs']))
    print('-' * 10)

    alexnet.train()
    
    if epoch % 2 == 0:
        for i, (name, param) in enumerate(alexnet.module.features.named_parameters()):
            if ('0.' in name) or ('6.' in name) or ('10.' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True        

        for i, (name, param) in enumerate(alexnet.module.classifier.named_parameters()):
            if ('4.' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True   
    else:
        for i, (name, param) in enumerate(alexnet.module.features.named_parameters()):
            if ('0.' in name) or ('6.' in name) or ('10.' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False        

        for i, (name, param) in enumerate(alexnet.module.classifier.named_parameters()):
            if ('4.' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False    
    
    training.pass_epoch(
        alexnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        #writer=writer
    )

    alexnet.eval()
    val_metrics = training.pass_epoch(
        alexnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        #writer=writer
    )
    
    if val_metrics[0].item() < val_loss:
        val_loss = val_metrics[0].item()
        print('Test set Accuracy Lowest Validation Loss:')
        training.pass_epoch(
                alexnet, loss_fn, test_loader,
                batch_metrics=metrics, show_running=True, device=device,
                #writer=writer
            )
        torch.save(alexnet.state_dict(), "alexnet_wYFCC.pth")

#writer.close()


# In[ ]:





# In[ ]:




