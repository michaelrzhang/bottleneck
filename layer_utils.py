from layers import *

# Template from Stanford CS231N class (Li, Karpathy, Johnson)

def affine_relu_forward_sub_mean(x, w, b):
  affine_out, fc_cache = affine_forward(x, w, b)
  mean_out = subtract_mean_forward(affine_out)
  out, relu_cache = relu_forward(mean_out)
  cache = (fc_cache, relu_cache)
  return out, cache

def affine_relu_backward_sub_mean(dout, cache):
  fc_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dx = subtract_mean_backward(dr)
  dx, dw, db = affine_backward(dx, fc_cache)
  return dx, dw, db


def affine_relu_forward_batchnorm(x, w, b, gamma, beta, bn_params, use_dropout = False, dropout_param = None):
  affine_out, fc_cache = affine_forward(x, w, b)
  bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_params)
  out, relu_cache = relu_forward(bn_out)
  if use_dropout:
    out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
  else:
    cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_relu_backward_batchnorm(dout, cache, use_dropout = False, dropout_param = None):
  if use_dropout:
    fc_cache, bn_cache, relu_cache, dropout_cache = cache
    dout = dropout_backward(dout, dropout_cache)
  else:
    fc_cache, bn_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dx, dgamma, dbeta = batchnorm_backward(dr, bn_cache)
  dx, dw, db = affine_backward(dx, fc_cache)
  return dx, dw, db, dgamma, dbeta

def affine_relu_forward(x, w, b, use_dropout = False, dropout_param = None):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  if use_dropout:
    out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
  else:
    cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache, use_dropout = False, dropout_param = None):
  """
  Backward pass for the affine-relu convenience layer
  """
  if use_dropout:
    fc_cache, relu_cache, dropout_cache = cache
    dout = dropout_backward(dout, dropout_cache)
  else:
    fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

