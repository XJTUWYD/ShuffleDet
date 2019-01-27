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
from nn_skeleton_imageNet import ModelSkeleton


def Tensor_shape(tensorname, tensorname_print):
  """print the shape of a tensor"""
  tensor_shape =  tensorname.shape.as_list()
  print('the shape of '+tensorname_print+' is: ' +str(tensor_shape))


class ClassName(object):
  """docstring for ClassName"""
  def __init__(self, arg):
    super(ClassName, self).__init__()
    self.arg = arg

class shuffleNet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self,mc)

      self._add_forward_graph()

  def _add_forward_graph(self):
    self.conv1 = self._conv_layer(
        'conv1', self.image_input, filters=24, size=3, stride=2, padding='SAME', freeze=False)
    self.pool1 = self._pooling_layer(
        'pool1', self.conv1 , size=2, stride=2, padding='SAME')
##########################ShuffleNet####################################################
    # #module d (repeat:1)
    self.stage_1_1 = self.shufflenetv2_module_d('stage_1_1', self.pool1,24, 64, 64, 64, 64)
    self.pool2 = self._pooling_layer('pool2', self.stage_1_1 , size=2, stride=2, padding='SAME')
    #module c(repeat:3)
    self.stage_2_1 = self.shufflenetv2_module_c('stage_2_1', self.pool2)
    self.stage_2_2 = self.shufflenetv2_module_c('stage_2_2', self.stage_2_1)
    self.stage_2_3 = self.shufflenetv2_module_c('stage_2_3', self.stage_2_2)
    self.pool3 = self._pooling_layer('pool3', self.stage_2_3 , size=2, stride=2, padding='SAME')
    #module d (repeat:1)
    self.stage_3_1 = self.shufflenetv2_module_d('stage_3_1', self.pool3, 128, 128, 128, 128, 128)
    # module c(repeat:7)
    self.stage_4_1 = self.shufflenetv2_module_c('stage_4_1', self.stage_3_1)
    self.stage_4_2 = self.shufflenetv2_module_c('stage_4_2', self.stage_4_1)
    self.stage_4_3 = self.shufflenetv2_module_c('stage_4_3', self.stage_4_2)
    self.stage_4_4 = self.shufflenetv2_module_c('stage_4_4', self.stage_4_3)
    self.stage_4_5 = self.shufflenetv2_module_c('stage_4_5', self.stage_4_4)
    self.stage_4_6 = self.shufflenetv2_module_c('stage_4_6', self.stage_4_5)
    self.stage_4_7 = self.shufflenetv2_module_c('stage_4_7', self.stage_4_6)
    #module d(repeat:1)
    self.stage_5_1 = self.shufflenetv2_module_d('stage_5_1', self.stage_4_7, 256, 256, 256, 256, 256)
    #module c(repeat:3)
    self.stage_6_1 = self.shufflenetv2_module_c('stage_6_1',self.stage_5_1)
    self.stage_6_2 = self.shufflenetv2_module_c('stage_6_2',self.stage_6_1)
    self.stage_6_3 = self.shufflenetv2_module_c('stage_6_3',self.stage_6_2)
    self.conv3 = self._conv_layer('conv3', self.stage_6_3 , filters=1000, size=3, stride=1,padding='SAME', xavier=False, relu=False, stddev=0.0001)
    self.result = self._global_average_pooling('global_average_pooling', self.conv3, stride = 1)
    Tensor_shape(self.result, 'result_shape')

  def shuffle_unit(self,x, groups):
      with tf.variable_scope('shuffle_unit'):
          n, h, w, c = x.get_shape().as_list()
          x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
          x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
          x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
      return x

  def shufflenetv2_module_c(self, layer_name, input_tensor,stddev=0.01, freeze=False):
    channels = int(input_tensor.get_shape()[3])
    half_channel = int(channels /2)
    top_tensor, bottom_tensor = tf.split(input_tensor, num_or_size_splits=2, axis=3)
    self.right_a = self._conv_bn_relu_layer(layer_name+'/right_a', bottom_tensor, filters = half_channel, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
    self.right_b = self._depthwise_conv_bn_layer(layer_name+'/right_b', self.right_a, filters = half_channel, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
    self.right_c = self._conv_bn_relu_layer(layer_name+'/right_c', self.right_b, filters = half_channel, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
    output = tf.concat([top_tensor, self.right_c], 3, name=layer_name+'/concat_right_left_1')
    return self.shuffle_unit(output,2)

  def shufflenetv2_module_d(self, layer_name,input_tensor,channels_left_a, channels_left_b,channels_right_a,channels_right_b,channels_right_c,stddev=0.01, freeze=False):
    channels = int(input_tensor.get_shape()[3])
    self.left_a = self._depthwise_conv_bn_layer(layer_name+'/left_a', input_tensor, filters = channels_left_a, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
    self.left_b = self._conv_bn_relu_layer(layer_name +'/left_b', self.left_a, filters = channels_left_b, size = 1, stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
    self.right_a = self._conv_bn_relu_layer(layer_name+'/right_a', input_tensor, filters = channels_right_a, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
    self.right_b = self._depthwise_conv_bn_layer(layer_name+'/right_b', self.right_a, filters = channels_right_b, size =3 , stride = 1, padding = 'SAME', stddev = stddev, freeze = freeze)
    self.right_c = self._conv_bn_relu_layer(layer_name+'/right_c', self.right_b, filters = channels_right_c, size = 1, stride =1 , padding = 'SAME', stddev = stddev ,freeze = freeze)
    output = tf.concat([self.left_b, self.right_c], 3, name=layer_name+'/concat_right_left_2')
    return self.shuffle_unit(output,2)


