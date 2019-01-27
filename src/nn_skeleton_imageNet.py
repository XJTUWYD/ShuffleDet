# Author: yadongwei (yadongwei2@gmail.com) frome XJTU
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from config import *


mc = imageNet_config()

def cabs(x):
    return tf.clip_by_value(x,-128,128)

def quantize(x, k):
    G = tf.get_default_graph()
    n = float(2**k)
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x * n) / n

def quantize_plus(x):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x) 

def _fw(x, bitW,force_quantization=False):
  G = tf.get_default_graph()
  if bitW == 32 and not force_quantization:
    return x
  if bitW == 1:   # BWN
    with G.gradient_override_map({"Sign": "Identity"}):
      E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
      return tf.sign(x / E) * E
        # x = tf.tanh(x)
        # x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
  x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0) # it seems as though most weights are within -1 to 1 region anyways
  return 2 * quantize(x, bitW) - 1
  # return quantize(x, bitW)

def _fa(x):
  return quantize_plus(x)


def quantize_active_overflow(x, bitA):
  if bitA == 32:
    return x
  return _quantize_overflow(x, bitA)


def quantize_weight_overflow(x, bitW):
  if bitW == 32:
    return x
  return _quantize_overflow(x, bitW)

def quantize_bias_overflow(x, bitW):
  if bitW == 32:
    return x
  return _quantize_overflow(x, bitW)


def _w_fold(w, gama, var, epsilon):
  """fold the BN into weight"""
  return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))


def _bias_fold(beta, gama, mean, var, epsilon):
  """fold the batch norm layer & build a bias_fold"""
  return tf.subtract(beta, tf.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))

def _quantize_overflow(x, k):
  """quantization of the weight and bias"""
  G = tf.get_default_graph()
  n = float(2**k - 1)
  max_value = tf.reduce_max(x)
  min_value = tf.reduce_min(x)
  with G.gradient_override_map({"Round": "Identity"}):
    step = tf.stop_gradient((max_value - min_value) / n)
  return tf.round((tf.maximum(tf.minimum(x, max_value), min_value) - min_value) / step) * step + min_value

def _add_loss_summaries(total_loss):
  """Add summaries for losses"""
  losses = tf.get_collection('losses')
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name, l)

def _variable_on_device(name, shape, initializer, trainable=True):
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_on_device_reuse(name, shape, initializer, trainable=True):
  dtype = tf.float32
  with tf.variable_scope('v_scope',reuse=tf.AUTO_REUSE) as scope1:
    if not callable(initializer):
      var = tf.get_variable('v_scope'+name, initializer=initializer, trainable=trainable)
    else:
      var = tf.get_variable(
          'v_scope'+name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def Tensor_shape(tensorname):
  """print the shape of a tensor"""
  tensor_shape =  tensorname.shape.as_list()
  pirnt('the  shape of '+str(tensorname)+'is' +str(tensor_shape))


def conv_layer(
   layer_name, inputs, filters, size, stride, padding='SAME',
    freeze=False, xavier=False, relu=True, stddev=0.001):
  use_pretrained_param = False
  with tf.variable_scope(layer_name) as scope:
    channels = inputs.get_shape()[3]
    if use_pretrained_param:
      if mc.DEBUG_MODE:
        print ('Using pretrained model for {}'.format(layer_name))
      kernel_init = tf.constant(kernel_val , dtype=tf.float32)
      bias_init = tf.constant(bias_val, dtype=tf.float32)
    elif xavier:
      kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      bias_init = tf.constant_initializer(0.0)
    else:
      kernel_init = tf.truncated_normal_initializer(
          stddev=stddev, dtype=tf.float32)
      bias_init = tf.constant_initializer(0.0)

    kernel = _variable_with_weight_decay(
        'kernels', shape=[size, size, int(channels), filters],
        wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

    biases = _variable_on_device('biases', [filters], bias_init,
                              trainable=(not freeze))
    conv = tf.nn.conv2d(
        inputs, kernel, [1, stride, stride, 1], padding=padding,
        name='convolution')
    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

    if relu:
      out = tf.nn.relu(conv_bias, 'relu')
    else:
      out = conv_bias

    return out

def conv_bn_wioutfusion_relu_layer(
      layer_name, inputs,  filters,
      size, stride, padding='SAME', freeze=False, relu=True, stddev=0.001):
  decay = 0.9
  with tf.variable_scope(layer_name) as scope:
    channels = inputs.get_shape()[3]
    kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    mean_val   = tf.constant_initializer(0.0)
    var_val    = tf.constant_initializer(1.0)
    gamma_val  = tf.constant_initializer(1.0)
    beta_val   = tf.constant_initializer(0.0)
    kernel = _variable_with_weight_decay(
        'kernels', shape=[size, size, int(channels), filters],
        wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
    conv = tf.nn.conv2d(
        inputs, kernel, [1, stride, stride, 1], padding=padding,
        name='convolution')
    gamma = _variable_on_device('gamma', [filters], gamma_val,
                                  trainable=(not freeze))
    beta  = _variable_on_device('beta', [filters], beta_val,
                                trainable=(not freeze))
    moving_mean  = _variable_on_device('moving_mean', [filters], mean_val, trainable=False)
    moving_variance   = _variable_on_device('moving_variance', [filters], var_val, trainable=False)
    conv = tf.nn.batch_normalization(
        conv, mean=moving_mean, variance=moving_variance, offset=beta, scale=gamma,
        variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')
    if relu:
      return tf.nn.relu(conv)
    else:
      return conv

def conv_bn_relu_layer(
    layer_name, inputs,  filters,
    size, stride, padding='SAME', freeze=False, relu=True, stddev=0.001):
  quantization = mc.QUANTIZATION
  decay = 0.9
  with tf.variable_scope(layer_name) as scope:
    channels = inputs.get_shape()[3]
    kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    mean_val   = tf.constant_initializer(0.0)
    var_val    = tf.constant_initializer(1.0)
    gamma_val  = tf.constant_initializer(1.0)
    beta_val   = tf.constant_initializer(0.0)
    kernel = _variable_with_weight_decay(
        'kernels', shape=[size, size, int(channels), filters],
        wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
    conv = tf.nn.conv2d(
        inputs, kernel, [1, stride, stride, 1], padding=padding,
        name='convolution')
    parameter_bn_shape = conv.get_shape()[-1:]
    gamma = _variable_on_device('gamma', parameter_bn_shape, gamma_val,
                                trainable=(not freeze))
    beta  = _variable_on_device('beta', parameter_bn_shape, beta_val,
                                trainable=(not freeze))
    moving_mean  = _variable_on_device('moving_mean', parameter_bn_shape, mean_val, trainable=False)
    moving_variance   = _variable_on_device('moving_variance', parameter_bn_shape, var_val, trainable=False)
    #fold weight and bias
    mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
    def mean_var_with_update():
      with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.identity(mean), tf.identity(variance)
    mean_1, var = mean_var_with_update()
    # mean_1, var = moving_mean, moving_variance
    w_fold = _variable_with_weight_decay(
        'w_fold', shape=[size, size, int(channels), 1],
        wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
    w_fold = _w_fold(kernel, gamma, var, mc.BATCH_NORM_EPSILON)
    bias_fold = _variable_on_device('bias_fold',[filters],beta_val,trainable = freeze)
    bias_fold = _bias_fold(beta, gamma, mean_1, var, mc.BATCH_NORM_EPSILON)
    kernel_quant = quantize_weight_overflow(w_fold, 8)
    inputs_quant = quantize_active_overflow(inputs,7)
    biases_quant = quantize_active_overflow(bias_fold,8)
    if quantization:
      conv_2 = tf.nn.conv2d(
          inputs_quant, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution_quantization')
      conv_2 =tf.nn.bias_add(conv_2, biases_quant)
    else:
      conv_2 = tf.nn.conv2d(inputs, w_fold, [1, stride, stride, 1], padding=padding, name='convolution')
      conv_2 =tf.nn.bias_add(conv_2, bias_fold)
    if relu:
      return tf.nn.relu(conv_2)
    else:
      return conv_2

def depthwise_conv_bn_layer(
    layer_name, inputs,  filters,
    size, stride, padding='SAME', freeze=False, relu=False, stddev=0.001):
  quantization = mc.QUANTIZATION
  decay = 0.9
  with tf.variable_scope(layer_name) as scope:
    channels = int(inputs.get_shape()[3])
    kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    mean_val   = tf.constant_initializer(0.0)
    var_val    = tf.constant_initializer(1.0)
    gamma_val  = tf.constant_initializer(1.0)
    beta_val   = tf.constant_initializer(0.0)
    kernel = _variable_with_weight_decay(
        'kernels', shape=[size, size, int(channels), 1],
        wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
    conv = tf.nn.depthwise_conv2d(
        inputs, kernel, [1, stride, stride, 1], padding=padding,
        name='convolution')
    gamma = _variable_on_device('gamma', [1], gamma_val,
                                trainable=(not freeze))
    beta  = _variable_on_device('beta', [1], beta_val,
                                trainable=(not freeze))
    moving_mean  = _variable_on_device('moving_mean', [filters], mean_val, trainable=False)
    moving_variance   = _variable_on_device('moving_variance', [filters], var_val, trainable=False)
    #fold weight and bias
    mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
    def mean_var_with_update():
      with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.identity(mean), tf.identity(variance)
    mean_1, var = mean_var_with_update()
    w_fold = _variable_with_weight_decay(
        'w_fold', shape=[size, size, int(channels), 1],
        wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
    bias_fold = _variable_on_device('bias_fold',[filters],beta_val,trainable = freeze)
    kernel = tf.transpose(kernel,[0,1,3,2])
    w_fold = _w_fold(kernel, gamma, var, mc.BATCH_NORM_EPSILON)
    w_fold = tf.transpose(w_fold,[0,1,3,2])
    bias_fold = _bias_fold(beta, gamma, mean_1, var, mc.BATCH_NORM_EPSILON)
    kernel_quant = quantize_weight_overflow(w_fold, 8)
    inputs_quant = quantize_active_overflow(inputs,7)
    biases_quant = quantize_active_overflow(bias_fold,8)
    if quantization:
      conv_2 = tf.nn.depthwise_conv2d(inputs_quant, kernel_quant, [1, stride, stride, 1], padding=padding,
       name='convolution_quantization_depth')
      conv_2 =tf.nn.bias_add(conv_2, biases_quant)
    else:
      conv_2 = tf.nn.depthwise_conv2d(inputs, w_fold, [1, stride, stride, 1], padding=padding, name='convolution')
      conv_2 =tf.nn.bias_add(conv_2, bias_fold)
    if relu:
      return tf.nn.relu(conv_2)
    else:
      return conv_2




def conv_depthwise(
    layer_name, inputs, filters, size, stride, padding='SAME',
    freeze=False, xavier=False, relu=True, stddev=0.001):
  use_pretrained_param = False
  with tf.variable_scope(layer_name) as scope:
    channels = inputs.get_shape()[3]
    # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
    # shape [h, w, in, out]
    if use_pretrained_param:
      if mc.DEBUG_MODE:
        print ('Using pretrained model for {}'.format(layer_name))
      kernel_init = tf.constant(kernel_val , dtype=tf.float32)
      # bias_init = tf.constant(bias_val, dtype=tf.float32)
    elif xavier:
      kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      # bias_init = tf.constant_initializer(0.0)
    else:
      kernel_init = tf.truncated_normal_initializer(
          stddev=stddev, dtype=tf.float32)
      # bias_init = tf.constant_initializer(0.0)

    kernel = _variable_with_weight_decay(
        'kernels', shape=[size, size, int(channels), 1],
        wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
    conv = tf.nn.depthwise_conv2d(
        inputs, kernel, [1, stride, stride, 1], padding=padding,
        name='convolution')
    if relu:
      out = tf.nn.relu(conv, 'relu')
    else:
      out = conv
    return out

def pooling_layer(
    layer_name, inputs, size, stride, padding='SAME'):
  with tf.variable_scope(layer_name) as scope:
    out =  tf.nn.max_pool(inputs,
                          ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)
    return out

def global_average_pooling(layer_name, inputs, stride, padding = 'VALID'):
  batch_num, height, width, channels = inputs.get_shape().as_list() 
  with tf.variable_scope(layer_name) as scope:
    out = tf.nn.avg_pool(inputs, 
                          ksize = [1, height, width, 1],
                          strides = [1,stride, stride,1],
                          padding = padding)
    out = tf.reduce_mean(out, [1,2])
    return out

def _fc_layer(
    self, layer_name, inputs, hiddens, flatten=False, relu=True,
    xavier=False, stddev=0.001):
  mc = self.mc

  use_pretrained_param = False
  if mc.LOAD_PRETRAINED_MODEL:
    cw = self.caffemodel_weight
    if layer_name in cw:
      use_pretrained_param = True
      kernel_val = cw[layer_name][0]
      bias_val = cw[layer_name][1]

  if mc.DEBUG_MODE:
    print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

  with tf.variable_scope(layer_name) as scope:
    input_shape = inputs.get_shape().as_list()
    if flatten:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs = tf.reshape(inputs, [-1, dim])
      if use_pretrained_param:
        try:
          # check the size before layout transform
          assert kernel_val.shape == (hiddens, dim), \
              'kernel shape error at {}'.format(layer_name)
          kernel_val = np.reshape(
              np.transpose(
                  np.reshape(
                      kernel_val, # O x (C*H*W)
                      (hiddens, input_shape[3], input_shape[1], input_shape[2])
                  ), # O x C x H x W
                  (2, 3, 1, 0)
              ), # H x W x C x O
              (dim, -1)
          ) # (H*W*C) x O
          # check the size after layout transform
          assert kernel_val.shape == (dim, hiddens), \
              'kernel shape error at {}'.format(layer_name)
        except:
          # Do not use pretrained parameter if shape doesn't match
          use_pretrained_param = False
          print ('Shape of the pretrained parameter of {} does not match, '
                 'use randomly initialized parameter'.format(layer_name))
    else:
      dim = input_shape[1]
      if use_pretrained_param:
        try:
          kernel_val = np.transpose(kernel_val, (1,0))
          assert kernel_val.shape == (dim, hiddens), \
              'kernel shape error at {}'.format(layer_name)
        except:
          use_pretrained_param = False
          print ('Shape of the pretrained parameter of {} does not match, '
                 'use randomly initialized parameter'.format(layer_name))

    if use_pretrained_param:
      if mc.DEBUG_MODE:
        print ('Using pretrained model for {}'.format(layer_name))
      kernel_init = tf.constant(kernel_val, dtype=tf.float32)
      bias_init = tf.constant(bias_val, dtype=tf.float32)
    elif xavier:
      kernel_init = tf.contrib.layers.xavier_initializer()
      bias_init = tf.constant_initializer(0.0)
    else:
      kernel_init = tf.truncated_normal_initializer(
          stddev=stddev, dtype=tf.float32)
      bias_init = tf.constant_initializer(0.0)

    weights = _variable_with_weight_decay(
        'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
        initializer=kernel_init)
    biases = _variable_on_device('biases', [hiddens], bias_init)
    self.model_params += [weights, biases]

    outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
    if relu:
      outputs = tf.nn.relu(outputs, 'relu')

    # count layer stats
    self.model_size_counter.append((layer_name, (dim+1)*hiddens))

    num_flops = 2 * dim * hiddens + hiddens
    if relu:
      num_flops += 2*hiddens
    self.flop_counter.append((layer_name, num_flops))

    self.activation_counter.append((layer_name, hiddens))

    return outputs
