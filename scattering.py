#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:42:54 2021

@author: SrAlejandro
"""



###############################################################################
# Finally, we import the `Scattering2D` class from the `kymatio.keras`
# package.
import tensorflow as tf
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from kymatio.sklearn import Scattering2D

import time


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_train_ = x_train[0:300]
y_train_ = y_train[0:300]
S = Scattering2D(J=2, shape=(28, 28))
#classifier = LogisticRegression(max_iter=20)
classifier = SVC()
estimators = [('scatter', S), ('clf', classifier)]
pipeline = Pipeline(estimators)


pipeline.fit(x_train_, y_train_)

x_test_ = x_test[0:300]
y_test_ = y_test[0:300]

pipeline.score(x_test, y_test)
#model.summary()


def Scattering(scatter_net, train_inp, train_target, test_inp, test_target):
    classifier = SVC()
    estimators = [('scatter', scatter_net), ('clf', classifier)]
    pipeline = Pipeline(estimators)
    
    start_time = time.time()
    pipeline.fit(train_inp, train_target)
    end_time = time.time()
    
    print('training time = ', end_time - start_time)
    
    print(pipeline.score(test_inp, test_target ))
    return pipeline
    









