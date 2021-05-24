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



def translate(image:torch.Tensor,u,v)-> torch.Tensor:
    points = image.nonzero()
    for point in points:
        if point[1] + u >28:
            return image
        elif point[2] + v > 28:
            return image
    image =torch.roll(image, shifts=(u, v), dims=(u, v))
    return image    



def rotate2(image, d):
    """Rotate the image by d/180 degrees."""
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
    """Rotate the image by d/180 degrees."""
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
    """Skew the image provided.

    Taken from StackOverflow:
    http://stackoverflow.com/a/33088550/4855984
    """
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



def elastic_transform(image, alpha=36, sigma=5, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    
    :param image: a 28x28 image
    :param alpha: scale for filter
    :param sigma: the standard deviation for the gaussian
    :return: distorted 28x28 image
    """
    image.numpy()
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)









