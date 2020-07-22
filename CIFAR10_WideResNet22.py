import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(42)

data_dir = '../CIFAR10_CNN/data/cifar10'

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

# PyTorch datasets
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

batch_size = 256

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
def conv_2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(in_channels=ni, out_channels=nf, 
                     kernel_size=ks, stride=stride, 
                     padding=ks//2, bias=False)

def bn_relu_conv(ni, nf):
    return nn.Sequential(nn.BatchNorm2d(ni), 
                         nn.ReLU(inplace=True), 
                         conv_2d(ni, nf))

class ResidualBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)
    
    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)

def make_group(N, ni, nf, stride):
    start = ResidualBlock(ni, nf, stride)
    rest = [ResidualBlock(nf, nf) for j in range(1, N)]
    return [start] + rest

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

class WideResNet(nn.Module):
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
        super().__init__()      
        # Increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]
        
        # Add groups of BasicBlock(increase channels & downsample)
        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i>0 else 1
            layers += make_group(N, n_channels[i], 
                                 n_channels[i+1], stride)
        
        # Pool, flatten & add linear layer for classification
        layers += [nn.BatchNorm2d(n_channels[3]), 
                   nn.ReLU(inplace=True), 
                   nn.AdaptiveAvgPool2d(1), 
                   Flatten(), 
                   nn.Linear(n_channels[3], n_classes)]
        
        self.features = nn.Sequential(*layers)
        
    def forward(self, x): return self.features(x)
    
def wrn_22(): 
    return WideResNet(n_groups=3, N=3, n_classes=10, k=6)

model = wrn_22()

from fastai.basic_data import DataBunch
from fastai.train import Learner
from fastai.metrics import accuracy

data = DataBunch.create(train_ds, valid_ds, bs=batch_size, path='./data/cifar10')
learner = Learner(data, model, loss_func=F.cross_entropy, metrics=[accuracy])
learner.clip = 0.1 # gradient is clipped to be in range of [-0.1, 0.1]

# Find best learning rate
learner.lr_find()
learner.recorder.plot() # select lr with largest negative gradient (about 5e-3)

# Training
epochs = 1
lr = 5e-3
wd = 1e-4

import time

t0 = time.time()
learner.fit_one_cycle(epochs, lr, wd=wd) # wd is the lambda in l2 regularization
t1 = time.time()

print('time: ', t1-t0)

# training process diagnostics
learner.recorder.plot_lr()
learner.recorder.plot_losses()
learner.recorder.plot_metrics()

torch.save(model.state_dict(), 'cifar10-wrn22.pth')

hyper_params = {
	'arch': str(model),
    'num_epochs': epochs,
    'opt_func': 'Adam',
    'batch_size': batch_size,
    'lr': lr,
    'scheduler': 'one-cycle',
    'weight_decay': wd,
    'grad_clip': learner.clip
}

'''
metrics = {
    'train_loss': history[-1].get('train_loss'),
	'val_acc': history[-1].get('val_acc'),
	'val_loss': history[-1].get('val_loss'),
	'time': t1-t0
}
'''

import json

with open('hyper_params.json', 'w') as fp:
    json.dump(hyper_params, fp)

'''
with open('metrics.json', 'w') as fp:
    json.dump(metrics, fp)
'''


