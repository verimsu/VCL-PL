"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
    
class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        super(AugmentedDataset, self).__init__()
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            try:
                self.waugmentation_transform = transform['waugment']
            except:
                self.waugmentation_transform = transform['standard']
        else:
            print('Transform is not a dictionary!')
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {}
        image, label = self.dataset.__getitem__(index)
        
        sample['image'] = self.image_transform(image)
        sample['image_waugmented'] = self.waugmentation_transform(image)
        sample['target'] = label

        return sample
    
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None, transform=None):
        super(NeighborsDataset, self).__init__()
        
        self.anchor_transform = transform['standard']
        self.neighbor_transform = transform['augment']
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        output['anchor'] = self.anchor_transform(anchor['image'])
        output['neighbor'] = self.anchor_transform(neighbor['image'])
        output['target'] = anchor['target']
        
        return output