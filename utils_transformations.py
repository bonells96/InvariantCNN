#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:15:14 2021

@author: SrAlejandro
"""
import numpy as np
import torch
from scipy.ndimage import affine_transform
from scipy.ndimage import geometric_transform
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates



def translate(image,u,v):
    points = image.nonzero()
    for point in points:
        if point[1] + u >28:
            return image
        elif point[2] + v > 28:
            return image
    image =torch.roll(image, shifts=(u, v), dims=(u, v))
    return image    



"""
def rotate2(image, d):
    d = torch.tensor(d)
    center = 0.5*torch.tensor(image.shape)
    rot = torch.tensor([[torch.cos(d), torch.sin(d)],[-torch.sin(d), torch.cos(d)]])
    offset = torch.matmul((center-torch.matmul(center, rot)), (torch.linalg.inv(rot)))
    affine_transform(image,rot,order=2, offset=-offset, cval=0.0,output=torch.float32)
    return affine_transform(
        image,
        rot,
        order=2,
        offset=-offset,
        cval=0.0,
        output=torch.float32)



def rotate(image, d):
    image.numpy()
    center = 0.5*np.array(image.shape)
    rot = np.array([[np.cos(d), np.sin(d)],[-np.sin(d), np.cos(d)]])
    offset = (center-center.dot(rot)).dot(np.linalg.inv(rot))
    rotate_image = affine_transform(
        image,
        rot,
        order=2,
        offset=-offset,
        cval=0.0,
        output=np.float32)
    return torch.from_numpy(rotate_image)



def skew(image):
    Skew the image provided.

    Taken from StackOverflow:
    http://stackoverflow.com/a/33088550/4855984
    
    image.numpy()
    image = image.reshape(28, 28)
    h, l = image.shape
    distortion = np.random.normal(loc=12, scale=1)

    def mapping(point):
        x, y = point
        dec = (distortion*(x-h))/h
        return x, y+dec+5
    return geometric_transform(
        image, mapping, (h, l), order=5, mode='nearest')


"""


def elastic_transformation(image, alpha, sigma):
    
    dx = gaussian_filter(np.random.uniform(high =1, low = -1, size=image.shape[0]), sigma, mode = 'constant', cval = 0)* alpha
    dy = gaussian_filter(np.random.uniform(high =1, low = -1, size=image.shape[1]), sigma, mode = 'constant', cval = 0)* alpha

    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1) ), np.reshape(y + dy, (-1, 1))
    image = torch.from_numpy(map_coordinates(image, indices, order=1).reshape(image.shape))
    return image

def translate2(image, u, v):
    
    #dx = gaussian_filter(np.random.uniform(high =1, low = -1, size=image.shape[0]), sigma, mode = 'constant', cval = 0)* alpha
    #dy = gaussian_filter(np.random.uniform(high =1, low = -1, size=image.shape[1]), sigma, mode = 'constant', cval = 0)* alpha

    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    indices = np.reshape(x + u*np.ones(image.shape[0]), (-1, 1) ), np.reshape(y + v*np.ones(image.shape[1]), (-1, 1))
    image = torch.from_numpy(map_coordinates(image, indices, order=1).reshape(image.shape))
    return image


def average_distance(image,samples, network, transformation, alpha, sigma, normalization = 1):
    s = 0
    feat_image = network(image.unsqueeze(0))
    for k in range(samples):
        image_deform = transformation(image.squeeze(0), alpha, sigma).unsqueeze(0)
        s += (feat_image - network(image_deform.unsqueeze(0))).norm()
    return s/(samples*normalization)


def average_distance_numpy(image, samples, network, transformation, alpha, sigma, normalization = 1):
    s = 0
    feat_image = network(image)
    for k in range(samples):
        image_deform= transformation(image, alpha, sigma)
        image_deform = image_deform.numpy()
        s+= np.linalg.norm((feat_image - network(image_deform)))
    return s/(samples*normalization)



def average_distance_images(image, images_comparison ,network):
    s = 0
    feat_image = network(image.unsqueeze(0))
    for k in range(len(images_comparison)):
        s += (feat_image - network(images_comparison[k].unsqueeze(0))).norm()
    return s/(len(images_comparison))
    



