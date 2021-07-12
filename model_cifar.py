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
    def __init__(self):
        super(CNN1_Cifar, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
        self.out = nn.Linear(in_features=5*5*64, out_features=10)
        
    def forward(self, x):
        x = x
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x,2,2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2,2)
        
            
            # First Linear layer
            
        x = self.out(x.reshape(-1, 5*5*64))
            
            #Output layer
        
            
        return x
    
############################################CNN Model version 2################################################
    

class CNN2_Cifar(nn.Module):
    def __init__(self):
        super(CNN2_Cifar, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=13)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 11)
        self.out = nn.Linear(in_features=10*10*64, out_features=10)
        
    def forward(self, x):
        x = x
            #First conv Layer
            
        x = F.relu(self.conv1(x))
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        
            
            # First Linear layer
            
        x = self.out(x.reshape(-1, 2*2*64))
            
            #Output layer
        
            
        return x
    
    
    
    
    
    
    