#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

training_acc = np.array(np.loadtxt('training_acc.txt', dtype='float32'))
training_dis = np.array(np.loadtxt('training_dis.txt', dtype='float32'))

print(training_acc.shape)
print(training_acc)
print(training_dis.shape)
print(training_dis)
