#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:17:12 2021

@author: SrAlejandro
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

import random

from scipy.fft import fft
from scipy.fft import fftfreq
from utils_plotting import plot_function
from utils_plotting import plot_2_functions
from utils_plotting import plot_3_functions
from utils_plotting import plot_3_functions_log
from utils_transformations import translate 
import seaborn as sns

############################Kernel functions Examples###############################
N= 300
x = np.linspace(-4, 4, 100)
kernel_map1 = np.exp((-(x-1)**2)/2)
kernel_map2 = np.exp((-(x + 1)**2)/2)
plot_2_functions(x, kernel_map1, kernel_map2,
                 'Representation of 2 points as Gaussian Functions','Map of x1', 'Map of x2')

plot_3_functions(x, kernel_map1, kernel_map2, kernel_map1*kernel_map2,
                 'Representation and kernel of 2 points', 'Map of x1', 'Map of x2', 'k(x1,x2)')
##############################Example Pooling#####################################
sns.set()

random.seed(10)
f=3
nums = np.linspace(0,1,N)
x = 30*np.random.normal(0,1, N) + 35*np.sin(2*np.pi* f* nums)
 #100*np.random.randn(N).cumsum() + 500*np.sin(2*np.pi* f* nums)+ 30*np.sin(2*np.pi* 0.5*f* nums)
pooling_function1 = gaussian_filter1d(x, sigma = 2) 
pooling_function2 = gaussian_filter1d(x, sigma = 10) 
plot_3_functions(np.linspace(0,1,N), x, pooling_function1, pooling_function2, '','Original Signal', 'Pooling with sigma=2', 'Pooling with sigma = 10')

xf = fftfreq(N, 1/N)[:N//2]



#plot_3_functions(xf, fft(x), fft(pooling_function1), fft(pooling_function2), '','Original Signal', 'Pooling with sigma=2', 'Pooling with sigma = 10')
plot_3_functions(xf, 2.0/N *np.abs(fft(x)[0:N//2]), 2.0/N *np.abs(fft(pooling_function1)[0:N//2]), 2.0/N *np.abs(fft(pooling_function2)[0:N//2]), '','Original Signal', 'Pooling with sigma=2', 'Pooling with sigma = 10')



plot_3_functions_log(xf, 2.0/N *np.abs(fft(x)[0:N//2]), 2.0/N *np.abs(fft(pooling_function1)[0:N//2]), 2.0/N *np.abs(fft(pooling_function2)[0:N//2]), '','Original Signal', 'Pooling with sigma=2', 'Pooling with sigma = 10')

