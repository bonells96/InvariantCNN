#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:34:45 2021

@author: SrAlejandro
"""
import torch
import torch.nn.functional as F
import matplotlib. pyplot as plt
from scipy.ndimage import gaussian_filter

import numpy as np
import itertools

########################################################################

def plot_function(x, y, title, label):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, y, color = 'blue', label =  label)
    ax.legend()
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('X axis', fontsize = 16)
    ax.set_ylabel('Y axis', fontsize = 16)
    plt.show()


########################################################################

def plot_2_functions(x, y1, y2, title, label1, label2):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, y1, color = 'blue', label =  label1)
    ax.plot(x, y2, color = 'green', label = label2)
    ax.legend()
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('X axis', fontsize = 16)
    ax.set_ylabel('Y axis', fontsize = 16)
    plt.show()

########################################################################

def plot_3_functions(x, y1, y2, y3, title, label1, label2, label3):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, y1, color = 'blue', label =  label1)
    ax.plot(x, y2, color = 'red', label = label2)
    ax.plot(x, y3, color = 'green', label = label3)
    ax.legend()
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('X axis', fontsize = 16)
    ax.set_ylabel('Y axis', fontsize = 16)
    plt.show()
    
########################################################################

def plot_images(images):
    images = images = torch.cat([i.unsqueeze(0) for i in images], dim = 0).cpu()
    n_images = images.shape[0]
    fig = plt.figure(figsize = (20,5))
    
    for i in range(n_images):
        ax = fig.add_subplot(1, n_images, i+1)
        ax.imshow(images[i].squeeze(0), cmap = 'bone')
        ax.set_title('Digits')
        ax.axis('off')
        




def plot_filter(images, filter):

    images = images = torch.cat([i.unsqueeze(0) for i in images], dim = 0).cpu()
    filter = torch.FloatTensor(filter).unsqueeze(0).unsqueeze(0).cpu()
    
    n_images = images.shape[0]

    filtered_images = F.conv2d(images, filter)

    fig = plt.figure(figsize = (20, 5))
    
    for i in range(n_images):

        ax = fig.add_subplot(2, n_images, i+1)
        ax.imshow(images[i].squeeze(0), cmap = 'bone')
        ax.set_title('Original')
        ax.axis('off')

        image = filtered_images[i].squeeze(0)

        ax = fig.add_subplot(2, n_images, n_images+i+1)
        ax.imshow(image, cmap='bone')
        ax.set_title(f'Filtered')
        ax.axis('off')
        
        
        
def plot_pooling(images, pool_size):

    images = torch.cat([i.unsqueeze(0) for i in images], dim = 0).cpu()
    
    #if pool_type.lower() == 'max':
    pool1 = F.max_pool2d
   # elif pool_type.lower() in  ['mean', 'avg']:
    pool2 = F.avg_pool2d
   # else:
   #    raise ValueError(f'pool_type must be either max or mean, got: {pool_type}')
    
    n_images = images.shape[0]

    #pooled_images_max = pool1(images, kernel_size = pool_size)
    pooled_images_max = gaussian_filter(images, 1)

    pooled_images_avg = pool2(images, kernel_size = pool_size)
    fig = plt.figure(figsize = (20, 10))
    
    for i in range(n_images):

        ax = fig.add_subplot(3, n_images, i+1)
        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title('Original')
        ax.axis('off')

        image = pooled_images_max[i].squeeze(0)

        ax = fig.add_subplot(3, n_images, n_images+i+1)
        ax.imshow(image, cmap='bone')
        ax.set_title('Gauss Filter')
        ax.axis('off')
        
        image = pooled_images_avg[i].squeeze(0)
        
        ax = fig.add_subplot(3, n_images, 2*n_images+i+1)
        ax.imshow(image, cmap='bone')
        ax.set_title('Avg Pool')
        ax.axis('off');
    
    
def Plot_conf_matrix(conf_matrix, target_names, title, accuracy):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    misclass = 1 - accuracy
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, "{:,}".format(conf_matrix[i, j]),
                     horizontalalignment="center",
                     color="red")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    
