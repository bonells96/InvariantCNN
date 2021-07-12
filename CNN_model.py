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
import time


############################################CNN Model version 1################################################

class CNN_version_1(nn.Module):
    def __init__(self, data_cifar = False):
        super(CNN_version_1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
        
        self.fc1 = nn.Linear(in_features=64*2*2, out_features=50)
        self.out = nn.Linear(in_features= 50, out_features= 10)
        
    def forward(self, x):
        x = x
            #First conv Layer
            
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, 4, 4)
            
            # First Linear layer
            
        x = F.elu(self.fc1(x.reshape(-1, 64*2*2)))
            
            #Output layer
        
        x = (self.out(x))
            
        return x

############################################CNN Model Feature Maps################################################

class CNN_version_feat_map(nn.Module):
    def __init__(self):
        super(CNN_version_feat_map, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
                
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.elu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
        x = F.avg_pool2d(x, 4, 4)
            
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
        total_size += label.shape[0]
    return (total_correct/total_size)*100
        

############################################Training Function####################################################


def training_Model(model, train_loader, learning_rate, number_epochs):
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(number_epochs):
        start_time = time.time()
        #losses = torch.zeros(number_epochs)
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
            #losses[epoch] = total_loss
        end_time = time.time()
        if (epoch%5)==0:
            print('Epoch :',epoch, "loss:", total_loss, " correct preds:", total_correct, 'time epoch :',end_time - start_time)
            print(accuracy(train_loader, model))
            
    #return losses

        
class CNN_version_2(nn.Module):
    def __init__(self):
        super(CNN_version_2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
        self.out = nn.Linear(in_features=64*2*2, out_features=10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.elu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
        x = F.avg_pool2d(x, 5, 5)                 
            
            # First Linear layer
            
        x = self.out(x.reshape(-1, 2*2*64))
            
            #Output layer
            
        
        return x
    


class CNN_version_2_featmap(nn.Module):
    def __init__(self):
        super(CNN_version_2_featmap, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 5, 5)                 
            
            # First Linear layer
                        
            #Output layer
            
        
        return x



class CNN_version_3(nn.Module):
    def __init__(self):
        super(CNN_version_3, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
        self.out = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.elu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
        x = F.avg_pool2d(x, 10, 10)                 
                                    
            #Output layer
        x = self.out(x.reshape(-1,64))            
        
        return x 
        

   
class CNN_version_3_featmap(nn.Module):
    def __init__(self):
        super(CNN_version_3_featmap, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3)
        
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 10, 10)                 
            
            # First Linear layer
                        
            #Output layer
            
        
        return x 
    
    
class CNN_version_4(nn.Module):
    def __init__(self):
        super(CNN_version_4, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
        
        self.out = nn.Linear(4*4*64, 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.elu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)                 
                                    
            #Output layer
        x = self.out(x.reshape(-1,4*4*64))            
        
        return x 
    
    
#########################################################################################################
    
    
class CNN_version_5(nn.Module):
    def __init__(self):
        super(CNN_version_5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=10)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 10)
        
        self.out = nn.Linear(10*10*64, 10)
        
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.relu(self.conv1(x))
            
            #Second conv Layer
            
        x = F.relu(self.conv2(x))
                                    
            #Output layer
        x = self.out(x.reshape(-1,10*10*64))            
        
        return x 
        

class CNN_version_5_featmap(nn.Module):
    def __init__(self):
        super(CNN_version_5_featmap, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 5)
                
    def forward(self, x):
        x = x
            
            #First conv Layer
            
        x = F.elu(self.conv1(x))
            
            #Second conv Layer
            
        x = F.elu(self.conv2(x))
                                    
            #Output layer        
        return x 









