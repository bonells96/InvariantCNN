#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:40:49 2021

@author: SrAlejandro
"""
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import DataLoaders
import random
import model_cifar
import ResNet

import matplotlib.pyplot as plt

from CNN_model import training_Model
from CNN_model import accuracy

import numpy as np

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset_cif = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

testset_cif = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)



#########################################################################################################
image, label = trainset_cif[1]

def imshow(img):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()
  
imshow(torchvision.utils.make_grid(image))

#########################################################################################################

random.seed(1)
num_samples_train = 5000
num_samples_test = 10000
learning_rate = 1e-5
number_epochs = 3000

index_train, index_test = DataLoaders.random_index(trainset_cif, testset_cif, num_samples_train, num_samples_test)
trainloader_torch, testloader_torch = DataLoaders.Data_load(index_train, index_test, trainset_cif, testset_cif,64)



#########################################################################################################
#########################################################################################################

model = model_cifar.CNN1_Cifar()
model_ResNet = ResNet.ResNet(ResNet.ResidualBlock, [2,2,2], 3, 10, 8)




training_Model(model, trainloader_torch, learning_rate, number_epochs)
training_Model(model_ResNet, trainloader_torch, learning_rate, number_epochs)

accuracy(testloader_torch, model)
accuracy(testloader_torch, model_ResNet)




