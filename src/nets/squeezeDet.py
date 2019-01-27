# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

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
from nn_skeleton import ModelSkeleton


class ClassName(object):
  """docstring for ClassName"""
  def __init__(self, arg):
    super(ClassName, self).__init__()
    self.arg = arg
    
    
class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
    self.conv1 = self._conv_weightonly_layer(
        'conv1', self.image_input, filters=64, size=3, stride=1,
        padding='SAME', freeze=True)
    self.pool1 = self._pooling_layer(
        'pool1', self.conv1 , size=2, stride=2, padding='SAME')
    shape_pool1 = self.pool1.shape.as_list()
    print(shape_pool1)
    self.pool1_1 = self._pooling_layer(
        'pool1_1', self.pool1, size=2, stride=2, padding='SAME')
    self.fire2 = self._compressed_fire_layer(
        'fire2', self.pool1_1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self.pool3 = self._pooling_layer(
        'pool3', self.fire2, size=2, stride=2, padding='SAME')

    self.fire4 = self._compressed_fire_layer(
        'fire4', self.pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self.pool5 = self._pooling_layer(
        'pool5', self.fire4, size=2, stride=2, padding='SAME')
    self.fire11 = self._compressed_fire_layer(
        'fire11', self.pool5, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    dropout11 = tf.nn.dropout(self.fire11, self.keep_prob, name='drop11')

    # self.conv1 = self._conv_weightonly_layer(
    #   'conv1', self.image_input, filters=32, size=3, stride=1,padding='SAME', freeze=False)
    # self.pool1 = self._pooling_layer(
    #   'pool1', self.conv1 , size=2, stride=2, padding='SAME')
    # self.pool1_1 = self._pooling_layer(
    #   'pool1_1', self.pool1, size=2, stride=2, padding='SAME')
    # self.fire2 = self._compressed_fire_layer(
    #   'fire2', self.pool1_1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    # self.pool4 = self._pooling_layer(
    #   'pool4', self.fire2, size=2, stride=2, padding='SAME')
    # self.fire5 = self._compressed_fire_layer(
    #   'fire5', self.pool4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    # self.pool8 = self._pooling_layer(
    #   'pool8', self.fire5, size=2, stride=2, padding='SAME')
    # self.fire9 = self._compressed_fire_layer(
    #   'fire9', self.pool8, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    # dropout11 = tf.nn.dropout(self.fire9, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)
    shape_preds = self.preds.shape.as_list()
    print('the shape of the predction:'+str(shape_preds))


  def _compressed_fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')


  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer_cp(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer_cp(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer_cp(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

  def _compressed_fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')


  def _Res_fire_layer(self, layer_name, inputs, rate_1x1, stddev=0.01,freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      Res fire layer operation.
    """
    sx1 = int(input_tensor.get_shape()[3])
    e1x1 = int(s1x1 * rate_1x1)
    e3x3 = s1x1 - e1x1
    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    concat_result = tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

    return concat_result + input_tensor


def res_module(self,  layer_name, input_tensor, stddev=0.01, freeze=False):
  """res layer constructor.
  Args:
    layer_name: layer name
    input_tensor: input tensor
    channels: number of 3x3 filters in first_layer of res_module.
    channels: number of 3x3 filters in second_layer of res_module.
    always we think f3x3 should be equal to s3x3, but there might be some differences, isn't it?
    freeze: if true, do not train parameters in this layer.
  Returns:
    res_module operation.
  """
  kernels = []
  biases = []

  channels = int(input_tensor.get_shape()[3])
  feature_map_of_firstlayer  = _conv_layer(layer_name+'/first_layer', input_tensor, filters=channels, size= 3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  feature_map_of_secondlayer = _conv_layer(layer_name+'/second_layer', feature_map_of_firstlayer, filters=channels, size=3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  return input_tensor + feature_map_of_secondlayer


