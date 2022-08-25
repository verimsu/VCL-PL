"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
from collections import OrderedDict
from data.celeba import CelebA
import torchvision

attrs_default = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

def get_criterion(p):
    if p['criterion'] == 'simclr':
        #from losses.losses import SimCLRLoss
        #criterion = SimCLRLoss(**p['criterion_kwargs'])
        from pytorch_metric_learning import losses
        criterion = losses.CosFaceLoss(num_classes=2, embedding_size=p['model_kwargs']['features_dim'])
        c_criterion = losses.IntraPairVarianceLoss()
    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))
    return criterion, c_criterion
    
def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048
    
    elif p['backbone'] == 'alexnet':
        return 4096

    else:
        raise NotImplementedError

def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet18':
        from models.resnet_cifar import resnet18
        backbone = resnet18()
    elif p['backbone'] == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50()
    elif p['backbone'] == 'alexnet':
        from models.alexnet import alexnet
        backbone = alexnet()
    from models.models import ClusteringModel
    model = ClusteringModel(backbone, **p['model_kwargs'])   
    return model


def get_train_dataset(p, transform, to_augmented_dataset=False, to_neighbors_dataset=False, YFCC = None):
    # Base dataset
    if p['train_db_name'] == 'CelebA' and YFCC == None:
        dataset = CelebA("/home/mehmetyavuz/datasets/CelebA128/", "/home/mehmetyavuz/datasets/list_attr_celeba.txt", 128, 'train', attrs_default)
    elif YFCC:
        #dataset = torchvision.datasets.ImageFolder(root='/truba_scratch/meyavuz/datasets/YFCC392K/', transform=transform)
        dataset = torchvision.datasets.ImageFolder(root='/home/mehmetyavuz/datasets/YFCC392K/', transform=transform)
    elif YFCC == 'train+YFCC':
        dataset_one = CelebA(root='/home/mehmetyavuz/datasets/', split='train', transform=transform)
        dataset_two = torchvision.datasets.ImageFolder(root='/home/mehmetyavuz/datasets/YFCC392K/', transform=transform)
        y_YFCC = np.load('y_YFCC.npy')
        dataset_two.targets = y_YFCC
        dataset = torch.utils.data.ConcatDataset([dataset_one, dataset_two])
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset, transform)
    elif to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'], transform)
    return dataset


def get_val_dataset(p, transform, to_augmented_dataset=False):
    # Base dataset
    if p['train_db_name'] == 'CelebA':
        dataset = CelebA("/truba_scratch/meyavuz/datasets/CelebA128/", "/truba_scratch/meyavuz/datasets/list_attr_celeba.txt", 128, 'valid', attrs_default)
        #dataset = CelebA(root='/home/mehmetyavuz/datasets/', split='valid', transform=transform)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset, transform)
    elif to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper 
        transform = {}
        transform['standard'] = transforms.Compose([
            #transforms.Resize(p['img_size']),
            transforms.RandomResizedCrop(p['augmentation_kwargs']['crop_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
        
        transform['waugment'] = transforms.Compose([
            #transforms.Resize(p['img_size']),
            transforms.RandomResizedCrop(**p['waugmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['waugmentation_kwargs']['color_jitter'])
            ], p=p['waugmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['waugmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['waugmentation_kwargs']['normalize'])
        ])        
        
        transform['augment'] = transforms.Compose([
            #transforms.Resize(p['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])
        return transform
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))

def get_val_transformations(p, dictionary = True):
    if dictionary:
        transform = {}
        transform['standard'] = transforms.Compose([
                #transforms.Resize(p['img_size']),
                transforms.CenterCrop(p['augmentation_kwargs']['crop_size']),
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation_kwargs']['normalize'])
            ])
    else:
        return transforms.Compose([
                #transforms.Resize(p['img_size']),
                transforms.CenterCrop(p['augmentation_kwargs']['crop_size']),
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation_kwargs']['normalize'])
            ])
    return transform

def get_optimizer(p, model):
    for param in model.backbone.parameters():
        param.requires_grad_(True)
    for param in model.cluster_head.parameters():
        param.requires_grad_(True)  
        
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
