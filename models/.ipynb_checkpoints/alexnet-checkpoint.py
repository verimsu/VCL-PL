"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models

def alexnet():
    backbone = models.__dict__['alexnet'](pretrained=True)
    backbone.classifier[6] = nn.Identity()
    return {'backbone': backbone, 'dim': 4096}
