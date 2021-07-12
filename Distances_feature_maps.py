#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:59:03 2021

@author: SrAlejandro
"""

import torch
import CNN_model
import ResNet

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import DataLoaders 

import utils_transformations
import matplotlib.pyplot as plt

import utils_plotting

transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])



trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

CNN = CNN_model.CNN_version_feat_map()
CNN2 = CNN_model.CNN_version_2_featmap()
CNN3 = CNN_model.CNN_version_3_featmap()
model3 = ResNet.Resnet_featmap(ResNet.ResidualBlock, [2,2,2], 1)



num_samples_train = 100
num_samples_test = 100

index_train, index_test = DataLoaders.random_index(trainset, testset, num_samples_train, num_samples_test)
trainloader_torch, testloader_torch = DataLoaders.Data_load(index_train, index_test, trainset, testset,64)

image1, target1 = trainset[0]

utils_plotting.plot_deformations_elastic(model3, image1)

utils_transformations.average_distance(image1, 100, model3, utils_transformations.elastic_transformation, 15, 10)



image1 = image1.unsqueeze(0)
a = CNN(image1)
plt.imshow(image1[0])
#CNN2(image)

image2, target2 = trainset[11]
image2 = image2.unsqueeze(0)
b = CNN(image2)
(a-b).norm()

print(target1, target2)


images = []
k=0
for image, label in trainset:
    if label == 4:
        images.append(image)
        k += 1
    if k > 50:
        break

a = CNN(images[0].unsqueeze(0))
s = 0
k=0
for image in images:
    b = CNN(image.unsqueeze(0))
    s += (a-b).norm()/a.norm()
    k+=1
print(s/(k-1))


shape = image1.shape
*shape

image2 = utils_transformations.elastic_transformation(image1[0],5,10)
plt.imshow(image2)
image3 = torch.from_numpy(image2)
plt.imshow(image1[0])









