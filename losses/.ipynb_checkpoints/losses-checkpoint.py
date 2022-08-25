import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
EPS=1e-8

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch

class vloss(nn.Module):
    def __init__(self):
        super(vloss,self).__init__()

    def softclip(self,tensor, mini):
        result_tensor = mini + F.softplus(tensor - mini)
        return result_tensor
        
    def forward(self, mu, logVar):
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # note the negative D_{KL} in appendix B of the paper
        s2q = torch.exp(logVar[:,1,:])
        s2p = torch.exp(logVar[:,0,:])
        sq = torch.exp(logVar[:,1,:]/2)
        sp = torch.exp(logVar[:,0,:]/2)
        
        KLD = -(self.softclip(torch.log(sq), -6) - self.softclip(torch.log(sp), -6) - (s2q + (mu[:,0,:] - mu[:,1,:])**2)/(2*s2p) + 0.5).mean()
        
        KLD += - 0.5 * torch.mean(1 + logVar - mu.pow(2) - logVar.exp())
        return KLD