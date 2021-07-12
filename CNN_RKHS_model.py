#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:12:01 2021

@author: SrAlejandro
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class CNN_version_2(nn.Module):
    def __init__(self):
        super(CNN_version_2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
        self.fc1 = nn.Linear(in_features=64, out_features=50)
        self.out = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 5, 5)                 
            
            # First Linear layer
            
        x = F.relu(self.fc1(x.reshape(-1, 64)))
            
            #Output layer
            
        x = self.out(x)
        
        return x

class CNN_version_2(nn.Module):
    def __init__(self):
        super(CNN_version_2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
        self.fc1 = nn.Linear(in_features=64, out_features=50)
        self.out = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 5, 5)                 
            
            # First Linear layer
            
        x = F.relu(self.fc1(x.reshape(-1, 64)))
            
            #Output layer
            
        
        return x













