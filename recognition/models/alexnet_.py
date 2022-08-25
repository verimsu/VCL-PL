# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = [ 'AlexNet_', 'alexnet_']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class Encoder(nn.Module):
    def __init__(self, hidden_n_1=4096, hidden_n_2=400):
        super(Encoder, self).__init__()
        self.interpreter = nn.Sequential(nn.Dropout(0.5),
                    nn.Linear(256 * 3 * 3, hidden_n_1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_n_1, hidden_n_1),
                    nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(hidden_n_1, hidden_n_2)
        self.fc_var  = nn.Linear(hidden_n_1, hidden_n_2)
        
    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        dist = torch.distributions.Normal(mu, std)
        paux = dist.rsample()
        return paux

    def forward(self, x):
        x = self.interpreter(x)
        rp = self.reparameterize(self.fc_mu(x), self.fc_var(x))
        return rp

class AlexNet_(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(AlexNet_, self).__init__()
        self.features = features
        self.encoder = Encoder()
        self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        return self.encoder(x)

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet_(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    model = AlexNet_(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
    return model
