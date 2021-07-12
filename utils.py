#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:34:45 2021

@author: SrAlejandro
"""

import matplotlib. pyplot as plt
plt.style.use('seaborn')


########################################################################

def plot_2_functions(x, y1, y2, title, label1, label2):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x, y1, color = 'blue', label =  label1)
    ax.plot(x, y2, color = 'red', label = label2)
    ax.legend()
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('X axis', fontsize = 16)
    ax.set_ylabel('Y axis', fontsize = 16)
    plt.show()

########################################################################

def plot_3_functions(x, y1, y2, y3, title, label1, label2, label3):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x, y1, color = 'blue', label =  label1)
    ax.plot(x, y2, color = 'yellow', label = label2)
    ax.plot(x, y3, color = 'green', label = label3)
    ax.legend()
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel('X axis', fontsize = 16)
    ax.set_ylabel('Y axis', fontsize = 16)
    plt.show()