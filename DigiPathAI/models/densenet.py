"""
	github cite: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import glob
import random

import imgaug
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


import numpy as np 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, BatchNormalization, Conv2D, MaxPooling2D,                             						AveragePooling2D, ZeroPadding2D, concatenate, 	
					Concatenate, UpSampling2D, Activation, Lambda)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics


# Densenet Model
bn_axis = 3
channel_axis = bn_axis

def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=bn_axis)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv

def dense_conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = dense_conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def unet_densenet121(input_shape, weights='imagenet'):
    blocks = [6, 12, 24, 16]
    n_channel = 3
    n_class = 2
    img_input = Input(input_shape + (n_channel,))
    
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    conv5 = x 
    
    conv6 = conv_block(UpSampling2D()(conv5), 320)
    conv6 = concatenate([conv6, conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128)
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    conv10 = conv_block(conv10, 64)
    res = Conv2D(n_class, (1, 1), activation='softmax')(conv10)
    model = Model(img_input, res)

    return model
#model = unet_densenet121(input_shape=(256,256), weights=None)
#model.summary()


# In[6]:

