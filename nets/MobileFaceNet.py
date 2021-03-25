# Copyright 2018 The AI boy xsr-ai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileFaceNets.

MobileFaceNets, which use less than 1 million parameters and are specifically tailored for high-accuracy real-time
face verification on mobile and embedded devices.

here is MobileFaceNets architecture, reference from MobileNet_V2 (https://github.com/xsr-ai/MobileNetv2_TF).

As described in https://arxiv.org/abs/1804.07573.

  MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices

  Sheng Chen, Yang Liu, Xiang Gao, Zhen Han

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

# Conv and InvResBlock namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# InvResBlock defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'ratio'])
DepthwiseConv = namedtuple('DepthwiseConv', ['kernel', 'stride', 'depth', 'ratio'])
InvResBlock = namedtuple('InvResBlock', ['kernel', 'stride', 'depth', 'ratio', 'repeate'])

def mobilenet_v2(inputs1,
                 inputs2,
                 is_training=False,
                 reuse=None,
                 scope='MultiModal_Net'):

  image_shape = inputs1.get_shape().as_list()
  voice_shape = inputs2.get_shape().as_list()
  if len(image_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(image_shape))

  with tf.variable_scope(scope, 'MultiModal_Net', [inputs1,inputs2], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        net_img = slim.conv2d(inputs1, 64, kernel_size=[1,1], stride=1, activation_fn=None, padding='VALID')
        net_vic = slim.conv2d(inputs2, 64, kernel_size=[1,1], stride=1, activation_fn=None, padding='VALID')
        with tf.variable_scope('Logits'):
            net = tf.concat((net_img, net_vic), axis=-1)
            logits = slim.conv2d(net, 128, kernel_size=[1,1], stride=1, activation_fn=None, padding='VALID')
            logits = tf.layers.flatten(logits)

  return logits

mobilenet_v2.default_image_size = 112

def prelu(input, name=''):
    alphas = tf.get_variable(name=name + 'prelu_alphas',initializer=tf.constant(0.25,dtype=tf.float32,shape=[input.get_shape()[-1]]))
    pos = tf.nn.relu(input)
    neg = alphas * (input - abs(input)) * 0.5
    return pos + neg


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'fused': True,
      'decay': 0.995,
      'epsilon': 2e-5,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  # Set weight_decay for weights in Conv and InvResBlock layers.
  #weights_init = tf.truncated_normal_initializer(stddev=stddev)
  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=prelu, normalizer_fn=slim.batch_norm): #tf.keras.layers.PReLU
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

def inference(images, audios, phase_train=False,
              weight_decay=0.00005, reuse=False):
    arg_scope = mobilenet_v2_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return mobilenet_v2(images, audios, is_training=phase_train, reuse=reuse)