#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:32:13 2021

@author: SrAlejandro
"""

#Libraries for Neural Networks
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms

#Library for Plotting
import matplotlib.pyplot as plt

#Import deformation functions
import utils_transformations
from utils_transformations import translate
from utils_transformations import rotate

from utils_plotting import plot_images, plot_pooling

from CNN_model import CNN_version_1, training_Model
import ResNet



batch_size = 64

################################Loading Data Sets################################################

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


################################Create a Transformed Datasets###########################################












#######################################################################################################

indices_train_set = torch.randperm(len(mnist_trainset))[:200]
indices_test_set = torch.randperm(len(mnist_testset))[:200]

train_dataset = torch.utils.data.Subset(mnist_trainset, indices_train_set)
test_dataset =  torch.utils.data.Subset(mnist_testset, indices_test_set)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)


####################################Training of a Model############################################

model = ResNet.ResNet(ResNet.ResidualBlock, [2,2,2], 1)
model
#mod = ScatterModel(81,7,7)

training_Model(model, train_dataloader, 1e-4, 1000)

from CNN_model import accuracy
accuracy(test_dataloader, model)


x = torch.randn((28, 28))


image, label = train_dataset[0]

images = []
k=0
for image, label in test_dataset:
    if label == 4:
        images.append(image)
        k += 1
    if k > 4:
        break

plot_images(images)
plot_pooling(images, 2)
#####################################################################################################



#####################################################################################################
len(mnist_trainset)
images, labels = mnist_trainset[0:4]
print(image[0])
a = image.grad
print(a)
plt.imshow(image[0], cmap='gray')


image_deformed = utils_transformations.elastic_transform(image[0], alpha=50, sigma=5, random_state=None)
plt.imshow(image_deformed, cmap = 'gray')
plt.imshow(image[0])

image.nonzero()

a = torch.from_numpy(image_deformed)
plt.imshow(a)
