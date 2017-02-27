#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this is the main neural network 
"""
import numpy as np
import matplotlib.pyplot as plt # for plot images

from misc.data_utils import get_CIFAR10_data # load cifar-10 data
from layers.layers import * # basic layers
from layers.layer_utils import * # combined layers
from fastlayers.fast_layers import *
from cnnArc.cnn import *
from solver.solver import *
#from cs231n.solver import Solver
import time
from scipy.misc import imread, imresize
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray' # for gray image

def sanity_check():
	# load 3-layer cnn model
	# conv - relu - 2x2 max pool - affine - relu - affine - softmax
	model = ThreeLayerConvNet()
	N = 50
	X = np.random.randn(N, 3, 32, 32)
	y = np.random.randint(10, size=N) # N element, choose from 0~9

	loss, grads = model.loss(X, y)
	print 'initial loss: ', loss

	model.reg = 0.5
	loss, grads = model.loss(X, y)
	print 'initial loss (with reg): ', loss

def train_data(use_small_data=False):
	model = MyConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, use_bn=True) # define the network
	if use_small_data:
		num_train = 500
		small_data = {
			'X_train': data['X_train'][:num_train],
			'y_train': data['y_train'][:num_train],
			'X_val': data['X_val'],
			'y_val': data['y_val'],
		}
		solver = Solver(model, small_data, num_epochs=10, batch_size=50, update_rule='adam', optim_config={'learning_rate': 2e-3,}, verbose=True, print_every=10)
	else:
		solver = Solver(model, data, num_epochs=10, batch_size=100, update_rule='adam', optim_config={'learning_rate': 1e-3,}, verbose=True, print_every=10)
		
	solver.train()
	
	# show result
	plt.subplot(2, 1, 1)
	plt.plot(solver.loss_history, 'o')
	plt.xlabel('iteration')
	plt.ylabel('loss')

	plt.subplot(2, 1, 2)
	plt.plot(solver.train_acc_history, '-o')
	plt.plot(solver.val_acc_history, '-o')
	plt.legend(['train', 'val'], loc='upper left')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.show()
		
# load the CIFAR10 data
data = get_CIFAR10_data() # dict
# show data info
for k, v in data.iteritems():
	print '%s: ' %k, v.shape

train_data()
