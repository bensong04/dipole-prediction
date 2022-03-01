import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

def features(iterable):
    prod = 1
    for i in iterable:
        if type(prod) != int:
            raise ValueError("All elements of iterable must be integers")
        prod *= i
    return prod

class DipoleModel(nn.Module):
    def __init__(self, size):
        f = features(size)
        super(DipoleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_stack = nn.Sequential( \
            nn.Conv3d(3, 2), \
            nn.ReLU(), \
            nn.Dropout(), \
            nn.Conv3d(2, 1), \
            nn.ReLU(), \
            nn.Linear(f,int(f/3)),
            nn.Tanh(), \
            nn.Linear(int(f/3), int(f/9)), \
            nn.Tanh(), \
            nn.Linear(int(f/9), 1)
            )
        def forward(self, x):
            x = self.flatten(x)
            logits = self.conv_stack(x)
            return logits
