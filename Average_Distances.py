#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:41:07 2021

@author: SrAlejandro
"""

######Import Libraries of Models

from ResNet import Resnet_featmap
from ResNet import ResidualBlock
from CNN_model import CNN_version_2_featmap
from CNN_model import CNN_version_feat_map
from CNN_model import CNN_version_3_featmap
from CNN_model import CNN_version_5_featmap
from kymatio.sklearn import Scattering2D

######Import Functions for deformations

from utils_transformations import elastic_transformation
from utils_transformations import average_distance
from utils_transformations import average_distance_numpy
from utils_transformations import average_distance_images
from utils_transformations import translate2
######Import Functions for Plotting

from utils_plotting import plot_images


######Import Libraries for Loading Data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tensorflow.keras.datasets.mnist as datasetmnist
import torch

from random import seed
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


###########################################Load Data######################################################
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

(x_train, y_train), _= datasetmnist.load_data()
###########################################################################################################

#seed(1)
index = randint(0,200)
image, label = trainset[index]
image_tf = x_train[index]
label_tf = y_train[index]

############################################Plot Elastic Transformation######################################
image,_ = trainset[11]

images = [image, elastic_transformation(image.squeeze(0), 15 ,3).unsqueeze(0), elastic_transformation(image.squeeze(0), 22 ,3).unsqueeze(0), elastic_transformation(image.squeeze(0), 30, 3).unsqueeze(0)]
plot_images(images)
###########################################Create Models######################################################
model_CNN_1 = CNN_version_feat_map()
model_CNN_2 = CNN_version_2_featmap()
model_CNN_3 = CNN_version_3_featmap()
model_CNN_4 = CNN_version_5_featmap()
model_ResNet = Resnet_featmap(ResidualBlock, [2,2,2], 1)
S = Scattering2D(J=2, shape=(28, 28))
##############################################################################################################


#################Compute Normalization Constants -> The 
indexes = np.random.randint(low = 0, high = 5000, size = 2000)
samples = torch.utils.data.Subset(trainset, indexes)
images = []
labels = []
for sample in samples:
    image, label = sample
    images.append(image)
    labels.append(label)

norm_constant_CNN_1 = average_distance_images(image, images, model_CNN_1)
norm_constant_CNN_2 = average_distance_images(image, images, model_CNN_2)
norm_constant_CNN_3 = average_distance_images(image, images, model_CNN_3)
norm_constant_CNN_4 = average_distance_images(image, images, model_CNN_4)
norm_constant_ResNet = average_distance_images(image, images[0:1000], model_ResNet)

images_tf = x_train[indexes]
labels_tf = y_train[indexes]
s=0
feat_map = S(image_tf)
for image_ in images_tf:
    s += np.linalg.norm(feat_map - S(image_))
norm_constant_scattering = s/len(indexes)
print(norm_constant_scattering)
##############################################################################################################
##########################################Compute Distances Elastic Def###########################################
##############################################################################################################
sns.set()

n = 110
alpha_vec = np.linspace(0,30,n) #
distances_CNN_1 = np.zeros(n)
distances_CNN_2 =np.zeros(n)
distances_CNN_3 =np.zeros(n)
distances_CNN_4 =np.zeros(n)

distances_Resnet =np.zeros(n)
distances_scattering = np.zeros(n)
k = 0
for alpha in alpha_vec:
    distances_CNN_1[k] = average_distance(image, 50, model_CNN_1, elastic_transformation, alpha, 3, norm_constant_CNN_1)
    distances_CNN_2[k] = average_distance(image, 50, model_CNN_2, elastic_transformation, alpha, 3, norm_constant_CNN_2)
    distances_CNN_3[k] = average_distance(image, 50, model_CNN_3, elastic_transformation, alpha, 3, norm_constant_CNN_3)
    distances_CNN_4[k] = average_distance(image, 50, model_CNN_4, elastic_transformation, alpha, 3, norm_constant_CNN_4)   
    distances_Resnet[k] = average_distance(image, 50, model_ResNet, elastic_transformation, alpha, 3, norm_constant_ResNet)
    distances_scattering[k] = average_distance_numpy(image_tf, 50, S, elastic_transformation, alpha, 3, norm_constant_scattering)
    k+=1
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(alpha_vec, distances_CNN_1, color = 'blue', label =  'CNN1')
ax.plot(alpha_vec, distances_CNN_2, color = 'red', label = 'CNN2')
ax.plot(alpha_vec, distances_CNN_3, color = 'black', label = 'CNN3')
ax.plot(alpha_vec, distances_CNN_4, color = 'gray', label = 'CNN4')

ax.plot(alpha_vec, distances_Resnet, color = 'green', label = 'Resnet')
ax.plot(alpha_vec, distances_scattering, color= 'yellow', label = 'Scattering')
ax.legend()
ax.set_title('Distance', fontsize = 16)
ax.set_xlabel('alpha', fontsize = 16)
ax.set_ylabel('average distance', fontsize = 16)
plt.show()

#################################################################################################################
##############################################Compute Distances of Translations########################################
##################################################################################################################  




#########################################################
###################Compute relative Average distance to other images for Scattering


