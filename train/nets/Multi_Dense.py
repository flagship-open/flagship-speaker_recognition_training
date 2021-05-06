from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

slim = tf.contrib.slim

def multi_dense(inputs1,
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

multi_dense.default_image_size = 112

def prelu(input, name=''):
    alphas = tf.get_variable(name=name + 'prelu_alphas',initializer=tf.constant(0.25,dtype=tf.float32,shape=[input.get_shape()[-1]]))
    pos = tf.nn.relu(input)
    neg = alphas * (input - abs(input)) * 0.5
    return pos + neg


def dense_arg_scope(is_training=True,
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
    arg_scope = dense_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return multi_dense(images, audios, is_training=phase_train, reuse=reuse)