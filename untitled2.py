#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:58:19 2021

@author: SrAlejandro
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import tensorflow as tf

from kymatio.sklearn import Scattering2D

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import DataLoaders

import CNN_model
import ResNet
import scattering

from utils_plotting import  Plot_conf_matrix

import matplotlib.pyplot as plt

import random


transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])


#############################################Load Datasets#################################################

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

image,_ = trainset[0]
plt.imshow(image, (1,2,0))
