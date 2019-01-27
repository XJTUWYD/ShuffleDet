# Author: Yadong Wei(yadongwei2@gmail.com) 04/01/2019

"""shufflenetv2 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton_imageNet import *


def Tensor_shape(tensorname, tensorname_print):
  """print the shape of a tensor"""
  tensor_shape =  tensorname.shape.as_list()
  print('the shape of '+tensorname_print+' is: ' +str(tensor_shape))


def shuffle_unit(x, groups):
  with tf.variable_scope('shuffle_unit'):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
    x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
  return x

def shufflenetv2_module_c( layer_name, input_tensor,stddev=0.01, freeze=False):
  channels = int(input_tensor.get_shape()[3])
  half_channel = int(channels /2)
  top_tensor, bottom_tensor = tf.split(input_tensor, num_or_size_splits=2, axis=3)
  right_a = conv_bn_relu_layer(layer_name+'/right_a', bottom_tensor, filters = half_channel, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
  right_b = depthwise_conv_bn_layer(layer_name+'/right_b', right_a, filters = half_channel, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
  right_c = conv_bn_relu_layer(layer_name+'/right_c', right_b, filters = half_channel, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
  output = tf.concat([top_tensor, right_c], 3, name=layer_name+'/concat_right_left_1')
  return shuffle_unit(output,2)

def shufflenetv2_module_d( layer_name,input_tensor,channels_left_a, channels_left_b,channels_right_a,channels_right_b,channels_right_c,stddev=0.01, freeze=False):
  channels = int(input_tensor.get_shape()[3])
  left_a = depthwise_conv_bn_layer(layer_name+'/left_a', input_tensor, filters = channels_left_a, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
  left_b = conv_bn_relu_layer(layer_name +'/left_b', left_a, filters = channels_left_b, size = 1, stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
  right_a = conv_bn_relu_layer(layer_name+'/right_a', input_tensor, filters = channels_right_a, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
  right_b = depthwise_conv_bn_layer(layer_name+'/right_b', right_a, filters = channels_right_b, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
  right_c = conv_bn_relu_layer(layer_name+'/right_c', right_b, filters = channels_right_c, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
  output = tf.concat([left_b, right_c], 3, name=layer_name+'/concat_right_left_2')
  return shuffle_unit(output,2)

def shuffleNet(image_input):
  conv1 = conv_layer('conv1', image_input, filters=24, size=3, stride=2, padding='SAME', freeze=False)
  pool1 = pooling_layer('pool1', conv1 , size=2, stride=2, padding='SAME')
##########################ShuffleNet####################################################
    # #module d (repeat:1)
  stage_1_1 = shufflenetv2_module_d('stage_1_1', pool1,24, 64, 64, 64, 64)
  pool2 = pooling_layer('pool2', stage_1_1 , size=2, stride=2, padding='SAME')
  #module c(repeat:3)
  stage_2_1 = shufflenetv2_module_c('stage_2_1', pool2)
  stage_2_2 = shufflenetv2_module_c('stage_2_2', stage_2_1)
  stage_2_3 = shufflenetv2_module_c('stage_2_3', stage_2_2)
  pool3 = pooling_layer('pool3', stage_2_3 , size=2, stride=2, padding='SAME')
  #module d (repeat:1)
  stage_3_1 = shufflenetv2_module_d('stage_3_1', pool3, 128, 128, 128, 128, 128)
  # module c(repeat:7)
  stage_4_1 = shufflenetv2_module_c('stage_4_1', stage_3_1)
  stage_4_2 = shufflenetv2_module_c('stage_4_2', stage_4_1)
  stage_4_3 = shufflenetv2_module_c('stage_4_3', stage_4_2)
  stage_4_4 = shufflenetv2_module_c('stage_4_4', stage_4_3)
  stage_4_5 = shufflenetv2_module_c('stage_4_5', stage_4_4)
  stage_4_6 = shufflenetv2_module_c('stage_4_6', stage_4_5)
  stage_4_7 = shufflenetv2_module_c('stage_4_7', stage_4_6)
  #module d(repeat:1)
  stage_5_1 = shufflenetv2_module_d('stage_5_1', stage_4_7, 256, 256, 256, 256, 256)
  #module c(repeat:3)
  stage_6_1 = shufflenetv2_module_c('stage_6_1',stage_5_1)
  stage_6_2 = shufflenetv2_module_c('stage_6_2',stage_6_1)
  stage_6_3 = shufflenetv2_module_c('stage_6_3',stage_6_2)
  conv3 = conv_layer('conv3', stage_6_3 , filters=1000, size=3, stride=1,padding='SAME', xavier=False, relu=False, stddev=0.0001)
  result = global_average_pooling('global_average_pooling', conv3, stride = 1)
  return result


