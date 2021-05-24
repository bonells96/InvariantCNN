#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:26:51 2021

@author: SrAlejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kymatio.torch import Scattering2D
import kymatio
###################################CNN class version 1#####################################################

class CNNv1(nn.Module):
    def __init__(self):
        super(CNNv1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
        
        self.fc1 = nn.Linear(in_features=64*2*2, out_features=50)
        self.out = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4, 4)
            
            # First Linear layer
            
        x = F.relu(self.fc1(x.reshape(-1, 64*2*2)))
            
            #Output layer
        
        x = F.softmax(self.out(x), dim=1)
            
        return x
        
#########################################CNN class version 2##################################################


class CNNv2(nn.Module):
    def __init__(self):
        super(CNNv2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size = 3)
        
        self.fc1 = nn.Linear(in_features=64, out_features=50)
        self.out = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)     
            
            #Third Conv Layer
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 4, 4)
            
            
            # First Linear layer
            
        x = F.relu(self.fc1(x.reshape(-1, 64)))
            
            #Output layer
            
        x = F.softmax(self.out(x), dim=1)
            
        return x



##############################################Scattering Network#########################################################


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)


scattering = Scattering2D(shape = (28, 28), J=2)

K=81
model = nn.Sequential(
    View(K, 7, 7),
    nn.BatchNorm2d(K),
    View(K * 7 * 7),
    nn.Linear(K * 7 * 7, 10)
)





class ScatterModel(nn.Module):
    def __init__(self,*args, scattering= None):
        super(ScatterModel, self).__init__()
        
        self.shape = args
        self.bn1 = nn.BatchNorm2d(args[0])
        self.out = nn.Linear(args[0]*args[1]*args[2], 10)
        
        if scattering == None:
            scattering = Scattering2D(shape = (28, 28), J=2)
        self.scattering = scattering
   

        
    def forward(self, x):
        
        #First Scattering
        x = self.scattering(x)
        x = self.bn1(x.view(self.args[0], self.args[1], self.args[2]))
        x = self.out(x.view(self.args[0]*self.args[1]*self.args[2],-1))
        return x
        
        


##################################################ResNet########################################################

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, Id_downsample = None):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2= nn.Conv2d(out_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.Id_downsample = Id_downsample
    
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.Id_downsample:
            identity = self.Id_downsample(x)
        out += identity
        out = F.relu(out)
        return out
        

class ResNet(nn.Module):
    def __init__(self, block, layers):
        
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool = nn.max_pool2d(2,2)
           
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        Id_downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            Id_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(Block(self.in_channels, out_channels, stride, Id_downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



#################################################################################################################
#######################################Function for training the model############################################
#################################################################################################################

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



def training_Model(model, train_loader, learning_rate, number_epochs):
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(number_epochs):
        total_loss = 0
        total_correct = 0
        for batch in train_loader:
            images, labels = batch 
            preds = model(images)
            loss = F.cross_entropy(preds, labels)
        
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        print('Epoch :',epoch, "loss:", total_loss, " correct preds:", total_correct)
        
        
        







