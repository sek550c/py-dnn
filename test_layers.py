#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this is a test of some neural network layers
"""
import numpy as np
import matplotlib.pyplot as plt # for plot images

from misc.data_utils import get_CIFAR10_data # load cifar-10 data
from layers.layers import * # neural network layers
# from fastlayers.fast_layers import conv_forward_fast, conv_backward_fast # fast layer
from fastlayers.fast_layers import *
#from cs231n.solver import Solver
import time
from scipy.misc import imread, imresize
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray' # for gray image

"""# load the CIFAR10 data
data = get_CIFAR10_data() # dict
for k, v in data.iteritems():
	print '%s: ' %k, v.shape
"""

# this function test the convolutional layer implemented in layers.py
def test_conv_layer():

	kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
	# kitten is wide, and puppy is already square
	d = kitten.shape[1] - kitten.shape[0]
	kitten_cropped = kitten[:, d/2:-d/2, :]

	img_size = 200   # Make this smaller if it runs too slow
	x = np.zeros((2, 3, img_size, img_size))
	x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1)) # 3*200*200
	x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

	# Set up a convolutional weights holding 2 filters, each 3x3
	w = np.zeros((2, 3, 3, 3))

	# The first filter converts the image to grayscale.
	# Set up the red, green, and blue channels of the filter.
	w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
	w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
	w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

	# Second filter detects horizontal edges in the blue channel.
	w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

	# Vector of biases. We don't need any bias for the grayscale
	# filter, but for the edge detection filter we want to add 128
	# to each output so that nothing is negative.
	b = np.array([0, 128])

	# Compute the result of convolving each input in x with each filter in w,
	# offsetting by b, and storing the results in out.
	time0 = time.clock()
	out_naive, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
	time1 = time.clock()
	out_fast, _ = conv_forward_fast(x, w, b, {'stride': 1, 'pad': 1})
	time2 = time.clock()
	print 'naive cost: %f' %(time1 - time0)
	print 'fast cost: %f' %(time2 - time1)
	print '%f speedup' %((time1 - time0)/(time2 - time1))
	
	'''
	def imshow_noax(img, normalize=True):
		""" Tiny helper to show images as uint8 and remove axis labels """
		if normalize:
		    img_max, img_min = np.max(img), np.min(img)
		    img = 255.0 * (img - img_min) / (img_max - img_min) # normalize
		plt.imshow(img.astype('uint8'))
		plt.gca().axis('off')
	'''
	
	# Show the original images and the results of the conv operation
	'''
	plt.subplot(2, 3, 1)
	imshow_noax(puppy, normalize=True)
	plt.title('Original image')
	plt.subplot(2, 3, 2)
	imshow_noax(out[0, 0])
	plt.title('Grayscale')
	plt.subplot(2, 3, 3)
	imshow_noax(out[0, 1])
	plt.title('Edges')
	plt.subplot(2, 3, 4)
	imshow_noax(kitten_cropped, normalize=False)
	plt.subplot(2, 3, 5)
	imshow_noax(out[1, 0])
	plt.subplot(2, 3, 6)
	imshow_noax(out[1, 1])
	plt.show()
	'''
	# show each channel
	'''
	plt.figure()
	plt.subplot(1, 3, 1)
	plt.imshow(puppy[:, :, 0])
	plt.subplot(1, 3, 2)
	plt.imshow(puppy[:, :, 1])
	plt.subplot(1, 3, 3)
	plt.imshow(puppy[:, :, 2])
	'''
	#plt.show()
def test_pool_layer():
	puppy = imread('puppy.jpg')
	print puppy.shape
	img_size = 200   # Make this smaller if it runs too slow
	x = np.zeros((1, 3, img_size, img_size))
	x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1)) # 3*200*200
	time0 = time.clock()
	out_naive, _ = max_pool_forward_naive(x, {'pool_height': 2, 'pool_width': 2, 'stride': 2})
	time1 = time.clock()
	out_fast, _ = max_pool_forward_fast(x, {'pool_height': 2, 'pool_width': 2, 'stride': 2})
	time2 = time.clock()
	print 'naive cost: %f' %(time1 - time0)
	print 'fast cost: %f' %(time2 - time1)
	print '%f speedup' %((time1 - time0)/(time2 - time1))
	
	# in order to show image, you have to change the shape a bit
	out1 = out_naive.reshape(3, img_size/2, img_size/2).transpose(1, 2, 0) 
	out2 = out_fast.reshape(3, img_size/2, img_size/2).transpose(1, 2, 0)
	
	def imshow_noax(img, normalize=True):
		if normalize:
			img_max, img_min = np.max(img), np.min(img)
			img = 255.0 * (img - img_min) / (img_max - img_min) # normalize
		plt.imshow(img.astype('uint8'))
		plt.gca().axis('off')
		
	plt.subplot(1, 3, 1)
	plt.imshow(puppy)
	plt.title('Original image')
	plt.subplot(1, 3, 2)
	plt.imshow(out1)
	plt.title('naive pool image')
	plt.subplot(1, 3, 3)
	plt.imshow(out2)
	plt.title('fast pool image')
	plt.show()
	
#test_conv_layer()
test_pool_layer()
