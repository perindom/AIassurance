import torch
import torch.nn as nn
import torch.nn.functional as F

### List of models for attribution generation

class SeedNetwork(nn.Module):
    """3 layer ANN for prediction of minimum number among C numbers"""
    def __init__(self, C=5, fc1=16, fc2=8):
        """ C represents number of classes """
        super(SeedNetwork, self).__init__()
        self.fc1 = nn.Linear(C, fc1)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc1, fc2)
        self.dp2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(fc2, C)
    
    def forward(self, x):
        """ Forward pass with input x of shape N x C """
        #Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp1(x)
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp2(x)
        #Output Layer
        x = torch.sigmoid(self.output(x))
        return x