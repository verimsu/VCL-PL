import torch
import torch.nn as nn
import torch.nn.functional as F

##############
class Encoder(nn.Module):
    def __init__(self, hidden_n_1, hidden_n_2):
        super(Encoder, self).__init__()
        self.interpreter = nn.Sequential(nn.Dropout(0.5),
                    nn.Linear(hidden_n_1, hidden_n_2),
                    nn.ReLU(inplace=True))
        self.fc = nn.Linear(hidden_n_2, hidden_n_2)

    def forward(self, x):
        x = self.interpreter(x)
        x = self.fc(x)
        return F.normalize(x, dim = 1)

##############
class ClusteringModel(nn.Module):
    def __init__(self, backbone, features_dim=128, num_heads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.num_heads = num_heads
        assert(isinstance(self.num_heads, int))
        assert(self.num_heads > 0)
        self.cluster_head = nn.ModuleList([Encoder(hidden_n_1=self.backbone_dim, hidden_n_2=features_dim) for _ in range(self.num_heads)])

    def forward(self, x):
        features = self.backbone(x)
        out = [cluster_head(features) for cluster_head in self.cluster_head]    
        return out