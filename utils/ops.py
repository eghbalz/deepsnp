#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C)
#   Software Competence Center Hagenberg GmbH (SCCH)
#   Institute of Computational Perception (CP) @ JKU Linz
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 17.09.2018 $
# by : fischer $
# Developers: Hamid Eghbalzadeh, Lukas Fischer

# --- imports -----------------------------------------------------------------
import keras.backend as K
from keras.layers import Activation, Conv2D, BatchNormalization, Dropout, AveragePooling2D, concatenate
from keras.regularizers import l2


def conv_block(ip, nb_filter, kernel_size=(1, 3), strides=(1, 1), padding='same', bias=False, init='he_uniform',
               dropout_rate=None, weight_decay=1E-4):
    """
    Apply Relu 1x3, Conv2D, optional dropout
    :param ip:              input keras tensor
    :param nb_filter:       number of filters
    :param kernel_size:     kernel size
    :param strides:         strides
    :param padding:         padding
    :param bias:            bias
    :param init:            init weights
    :param dropout_rate:    dropout rate
    :param weight_decay:    weight decay factor
    :return: keras tensor with relu, convolution2d and optional dropout added
    """

    x = Activation('relu')(ip)
    x = Conv2D(filters=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=bias,
               kernel_initializer=init, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, hidden_filter, hidden_dilation, strides=None, dropout_rate=None, init='he_uniform',
                     padding='same', bias=False, weight_decay=1E-4, block_idx=0, block_thresh=2):
    """
    Apply BatchNorm, Conv2D, optional dropout and average pooling
    :param ip:              keras tensor
    :param nb_filter:       number of filters
    :param hidden_filter:   number of hidden filters
    :param hidden_dilation: number of hidden dilations
    :param strides:         strides
    :param dropout_rate:    dropout rate
    :param padding:         padding
    :param bias:            bias
    :param init:            init weights
    :param weight_decay:    weight decay factor
    :param block_idx:       block index
    :param block_thresh:    conv layers with block_idx < block_thresh are run with hidden_filter/hidden_dilation params
    :return: keras tensor, after applying batch_norm, relu-conv, optional dropout, averagepool
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    if block_idx < block_thresh:
        x = Conv2D(nb_filter, kernel_size=hidden_filter, dilation_rate=(1, hidden_dilation[1] ** block_idx),
                   kernel_initializer=init, padding=padding, use_bias=bias, kernel_regularizer=l2(weight_decay))(ip)
    else:
        x = Conv2D(nb_filter, kernel_size=(1, 2), strides=(1, 2), kernel_initializer=init, padding=padding,
                   use_bias=bias, kernel_regularizer=l2(weight_decay))(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D(pool_size=(1, 2), strides=strides)(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    return x


def dil_transition_block(ip, nb_filter, hidden_filter, dropout_rate=None, init='he_uniform',
                         padding='same', bias=False, weight_decay=1E-4, block_idx=0, strides=(1, 2), dilation=3,
                         block_thresh=4):
    """
    Apply BatchNorm, dilated Conv2D, optional dropout and average pooling
    :param ip:              keras tensor
    :param nb_filter:       number of filters
    :param hidden_filter:   number of hidden filters
    :param dilation:        dilation rate
    :param strides:         strides
    :param dropout_rate:    dropout rate
    :param padding:         padding
    :param bias:            bias
    :param init:            init weights
    :param weight_decay:    weight decay factor
    :param block_idx:       block index
    :param block_thresh:    conv layers with block_idx < block_thresh are run with hidden_filter/hidden_dilation params
    :return: keras tensor, after applying batch_norm, relu-conv, optional dropout, averagepool
    :return:
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    if block_idx < block_thresh:
        # keeps dimension
        x = Conv2D(nb_filter, kernel_size=hidden_filter, dilation_rate=(1, dilation),
                   kernel_initializer=init, padding=padding, use_bias=bias, kernel_regularizer=l2(weight_decay))(ip)
    else:
        # reduces dimension
        x = Conv2D(nb_filter, kernel_size=(1, 2), strides=(1, 2), kernel_initializer=init, padding=padding,
                   use_bias=bias, kernel_regularizer=l2(weight_decay))(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((1, 2), strides=strides)(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """
    Build a dense_block where the output of each conv_block is fed to subsequent ones
    :param x:               keras tensor
    :param nb_layers:       the number of layers of conv_block to append to the model.
    :param nb_filter:       number of filters
    :param growth_rate:     growth rate
    :param dropout_rate:    dropout rate
    :param weight_decay:    weight decay factor
    :return: keras tensor with nb_layers of conv_block appended
    """

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, nb_filter=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        feature_list.append(x)
        x = concatenate(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


def attend_out(vects):
    """
    Output attention
    :param vects:
    :return:
    """
    soft, sig = vects  # (N, n_time, n_out), (N, n_time, n_out)
    sig = K.clip(sig, 1e-7, 1.)
    out = K.sum(soft * sig, axis=1) / K.sum(sig, axis=1)  # (N, n_out)
    return out


def attend_in(vects):
    """
    Input attention
    :param vects:
    :return:
    """
    soft, sig = vects  # (N, n_time, n_out), (N, n_time, n_out)
    out = soft * sig
    return out


def avg(soft):
    """
    Averaging
    :param soft:
    :return:
    """
    out = K.mean(soft, axis=1)
    return out


def outfunc(vects):
    cla, att = vects  # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)  # (N, n_out)
    return out
