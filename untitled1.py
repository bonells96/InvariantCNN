#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:12:04 2021

@author: SrAlejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



############################################CNN Model version 1################################################

class CNN_version_1(nn.Module):
    def __init__(self):
        super(CNN_version_1, self).__init__()
        
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
    
    
############################################Number correct classes ##############################################


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#########################################Accuracy########################################################

def accuracy(train_loader, model):
    
    total_correct = 0
    total_size = 0
    for batch in train_loader:
        
        images, label = batch
        preds = model(images)
        total_correct += get_num_correct(preds, label)
        total_size += label.size
    return (total_correct/total_size)*100
        

############################################Training Function####################################################


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
        print(accuracy(train_loader, model))
        
        
        
        
        
        