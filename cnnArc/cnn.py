import numpy as np

from layers.layers import *
from layers.layer_utils import *
from fastlayers.fast_layers import *


class MyConvNet(object):
  """
  A convolutional network with the following architecture:
  
  [conv - (bn) - relu - 2x2 max pool] * 2 - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), 
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0,
               dtype=np.float32, use_bn=True):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_bn = use_bn
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # params
    C, H, W = input_dim
    # init weights and bias
    conv_filter_num1 = 32 # 1st convolutional filter num
    conv_filter_num2 = 64 # 2nd convolutional filter num
    # conv layer 1
    W1 = weight_scale*np.random.randn(conv_filter_num1, C, 3, 3) # 32*3*3*3
    b1 = np.zeros((1, conv_filter_num1))
    if self.use_bn: # spatial batch norm
        gamma1 = np.ones((1, conv_filter_num1))
        beta1 = np.zeros((1, conv_filter_num1))
    # conv layer 2
    W2 = weight_scale*np.random.randn(conv_filter_num2, conv_filter_num1, 3, 3) # 64*32*3*3
    b2 = np.zeros((1, conv_filter_num2))
    if self.use_bn: # spatial batch norm
        gamma2 = np.ones((1, conv_filter_num2))
        beta2 = np.zeros((1, conv_filter_num2))
    # hidden affine layer
    W3 = weight_scale*np.random.randn(conv_filter_num2*H*W/16, hidden_dim) # assume size not change after conv
    b3 = np.zeros((1, hidden_dim))
    if self.use_bn: # bn
        gamma3 = np.ones((1, hidden_dim))
        beta3 = np.zeros((1, hidden_dim))
    # output affine layer
    W4 = weight_scale*np.random.randn(hidden_dim, num_classes)
    b4 = np.zeros((1, num_classes))
        
    if self.use_bn:
        self.params.update({'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3})
    else:
        self.params.update({'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    if self.use_bn:
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    # pass bn_param to the forward pass for the spatial bn layer
    if self.use_bn:
        mode = 'test' if y is None else 'train'
        bn_param1 = {'mode': mode}
        bn_param2 = {'mode': mode}
        bn_param3 = {'mode': mode}
    
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_bn:
        # conv -> batch norm -> relu -> max pool 1
        h1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param1, pool_param)
        # conv -> batch norm -> relu -> max pool 2
        h2, cache2 = conv_bn_relu_pool_forward(h1, W2, b2, gamma2, beta2, conv_param, bn_param2, pool_param)
        # affine -> batch norm -> relu
        h3, cache3 = affine_bn_relu_forward(h2, W3, b3, gamma3, beta3, bn_param3)
        # affine -> batch norm
        scores, cache4 = affine_forward(h3, W4, b4)
    else:
        # conv -> relu -> max pool 1
        h1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # conv -> relu -> max pool 2
        h2, cache2 = conv_relu_pool_forward(h1, W2, b2, conv_param, pool_param)
        # affine -> relu
        h3, cache3 = affine_relu_forward(h2, W3, b3)
        # affine 
        scores, cache4 = affine_forward(h3, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # softmax and loss/bp start
    loss, dscores = softmax_loss(scores, y)
    # regularization loss
    reg_loss = 0.5*self.reg*np.sum(W1*W1)
    reg_loss += 0.5*self.reg*np.sum(W2*W2)
    reg_loss += 0.5*self.reg*np.sum(W3*W3)
    reg_loss += 0.5*self.reg*np.sum(W4*W4)
    loss += reg_loss 
    
    if self.use_bn:
        # affine bp
        dh3, dW4, db4 = affine_backward(dscores, cache4)
        dW4 += self.reg*W4 # add regularization loss term
        # affine <- relu
        dh2, dW3, db3, dgamma3, dbeta3 = affine_bn_relu_backward(dh3, cache3)
        dW3 += self.reg*W3 # add regularization loss term
        # conv <- sbn <- relu <- max pool 2
        dh1, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_pool_backward(dh2, cache2)
        dW2 += self.reg*W2 # add regularization loss term
        # conv <- sbn <- relu <- max pool 1
        dx, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dh1, cache1)
        dW1 += self.reg*W1 # add regularization loss term
    else:
        # affine bp
        dh3, dW4, db4 = affine_backward(dscores, cache4)
        dW4 += self.reg*W4 # add regularization loss term
        # affine <- relu
        dh2, dW3, db3 = affine_relu_backward(dh3, cache3)
        dW3 += self.reg*W3 # add regularization loss term
        # conv <- relu <- max pool 2
        dh1, dW2, db2 = conv_relu_pool_backward(dh2, cache2)
        dW2 += self.reg*W2 # add regularization loss term
        # conv <- relu <- max pool 1
        dx, dW1, db1 = conv_relu_pool_backward(dh1, cache1)
        dW1 += self.reg*W1 # add regularization loss term
    # update params
    if self.use_bn:
        grads.update({'W1': dW1, 'b1': db1,'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'gamma1': dgamma1, 'gamma2': dgamma2, 'gamma3': dgamma3, 'beta1': dbeta1, 'beta2': dbeta2, 'beta3': dbeta3})
    else:
        grads.update({'W1': dW1, 'b1': db1,'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
