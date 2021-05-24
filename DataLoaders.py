#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:39:58 2021

@author: SrAlejandro
"""
import random
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import utils_transformations

random.seed(15)


#########################################Load Mnist DataSet#####################################################
#We load 2 instances of the class datasets that will be two tensors

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
################################################################################################################

def Data_Loaders(train_set, test_set, num_samples_train, num_samples_test):
    indices_train_set = torch.randperm(len(mnist_trainset))[:num_samples_train]
    indices_test_set = torch.randperm(len(mnist_testset))[:num_samples_test]

    train_dataset = torch.utils.data.Subset(mnist_trainset, indices_train_set)
    test_dataset =  torch.utils.data.Subset(mnist_testset, indices_test_set)
    return train_dataset, test_dataset

################################################################################################################

def random_index(trainset, testset, num_samples_train, num_samples_test):
    indices_train_set = torch.randperm(len(trainset))[:num_samples_train]
    indices_test_set = torch.randperm(len(testset))[:num_samples_test]
    return indices_train_set, indices_test_set


def Data_load(index_train, index_test, trainset, testset, batch_size):
    train_dataset = torch.utils.data.Subset(trainset, index_train)
    test_dataset =  torch.utils.data.Subset(testset, index_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)
    return train_dataloader, test_dataloader

################################################################################################################







################################################################################################################






