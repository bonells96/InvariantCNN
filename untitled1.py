#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:01:31 2021

@author: SrAlejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


############################################CNN Model version 1################################################

class CNN1_Cifar(nn.Module):
    def __init__(self, data_cifar = False):
        super(CNN1_Cifar, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.out = nn.Linear(in_features=64, out_features=10)
        
    def forward(self, x):
        x = x
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2,2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2,2)
            
            # First Linear layer
            
        x = self.out(x.reshape(-1, 64))
            
            #Output layer
        
            
        return x