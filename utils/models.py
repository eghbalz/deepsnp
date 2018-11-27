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
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, \
    Lambda, Reshape, LSTM, GRU, TimeDistributed, Bidirectional, Permute
from keras.layers.merge import Multiply
from keras.regularizers import l2

from utils.ops import dense_block, transition_block, dil_transition_block, attend_in, attend_out, avg


def BLVGG(input_tensor, first_filter, hidden_filter, first_dilation, num_classes):
    """
    Build the baseline VGG model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param first_dilation:  number of first dilations
    :param num_classes:     number of classes
    :return: keras tensor with nb_layers of conv_block appended
    """
    conv1 = Conv2D(filters=32, kernel_size=first_filter, strides=first_dilation, padding='valid',
                   name='input_conv1')(input_tensor)
    bn1 = BatchNormalization(name='bn1')(conv1)

    # Block 1
    x = Conv2D(64, hidden_filter, activation='relu', padding='valid', name='block1_conv1')(bn1)
    x = Conv2D(64, hidden_filter, activation='relu', padding='valid', name='block1_conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block1_pool')(x)
    x = Dropout(0.4)(x)

    # Block 2
    x = Conv2D(128, (1, 3), activation='relu', padding='valid', name='block2_conv1')(x)
    x = Conv2D(128, (1, 3), activation='relu', padding='valid', name='block2_conv2')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block2_pool')(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = Conv2D(256, (1, 3), activation='relu', padding='valid', name='block3_conv1')(x)
    x = Conv2D(256, (1, 3), activation='relu', padding='valid', name='block3_conv2')(x)
    x = Conv2D(256, (1, 3), activation='relu', padding='valid', name='block3_conv3')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block3_pool')(x)
    x = Dropout(0.2)(x)

    # Block 4
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block4_conv1')(x)
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block4_conv2')(x)
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block4_conv3')(x)
    x = BatchNormalization(name='bn5')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(x)
    x = Dropout(0.2)(x)

    # Block 5
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block5_conv1')(x)
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block5_conv2')(x)
    x = Conv2D(512, (1, 3), activation='relu', padding='valid', name='block5_conv3')(x)
    x = BatchNormalization(name='bn6')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block5_pool')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_classes, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax')(x)
    return predictions


def BLDenseNet(input_tensor, first_filter, hidden_filter, first_dilation, hidden_dilation, margin, dropout_rate,
               num_classes, depth=16, nb_dense_block=6, growth_rate=12, nb_filter=16, weight_decay=1E-4):
    """
    Build the baseline DenseNet model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param first_dilation:  number of first dilations
    :param hidden_dilation: number of hidden dilations
    :param margin:          data margin
    :param dropout_rate:    dropout rate
    :param num_classes:     number of classes
    :param depth:           number or layers
    :param nb_dense_block:  number of dense blocks to add to end
    :param growth_rate:     number of filters to add
    :param nb_filter:       number of filters
    :param weight_decay:    weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(filters=nb_filter, kernel_size=first_filter, strides=first_dilation, padding='valid',
               name='initial_conv2D', kernel_regularizer=l2(weight_decay))(input_tensor)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if margin <= 500 and block_idx < 3:
            strides = (1, 1)
        elif 500 < margin < 5000 and block_idx < 2:
            strides = (1, 1)
        else:
            strides = (1, 2)
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, hidden_filter=hidden_filter, hidden_dilation=hidden_dilation,
                             strides=strides, dropout_rate=dropout_rate, weight_decay=weight_decay, block_idx=block_idx)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = BatchNormalization(axis=concat_axis, epsilon=1.001e-5, name='bn')(x)

    x = Conv2D(num_classes, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)

    return out


def BLDilDenseNet(input_tensor, first_filter, hidden_filter, margin, dropout_rate, num_classes, depth=16,
                  nb_dense_block=6, growth_rate=12, nb_filter=16, weight_decay=1E-4):
    """
    Build the baseline dilation DenseNet model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param margin:          data margin
    :param dropout_rate:    dropout rate
    :param num_classes:     number of classes
    :param depth:           number or layers
    :param nb_dense_block:  number of dense blocks to add to end
    :param growth_rate:     number of filters to add
    :param nb_filter:       number of filters
    :param weight_decay:    weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # Initial convolution
    x = DilBLDenseNetFeats(input_tensor=input_tensor, hidden_filter=hidden_filter, first_filter=first_filter,
                           margin=margin, nb_filter=nb_filter, weight_decay=weight_decay, growth_rate=growth_rate,
                           dropout_rate=dropout_rate, nb_dense_block=nb_dense_block, depth=depth)

    x = BatchNormalization(axis=concat_axis, epsilon=1.001e-5, name='bn')(x)

    x = Conv2D(num_classes, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)

    return out


def BLLSTMDenseNet(input_tensor, first_filter, hidden_filter, first_dilation, hidden_dilation, margin, dropout_rate,
                   num_classes, depth=16, nb_dense_block=6, growth_rate=12, nb_filter=16, weight_decay=1E-4):
    """
    Build the baseline LSTM DenseNet model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param first_dilation:  number of first dilations
    :param hidden_dilation: number of hidden dilations
    :param margin:          data margin
    :param dropout_rate:    dropout rate
    :param num_classes:     number of classes
    :param depth:           number or layers
    :param nb_dense_block:  number of dense blocks to add to end
    :param growth_rate:     number of filters to add
    :param nb_filter:       number of filters
    :param weight_decay:    weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(filters=nb_filter, kernel_size=first_filter, strides=first_dilation, padding='valid',
               name='initial_conv2D', kernel_regularizer=l2(weight_decay))(input_tensor)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if margin <= 500 and block_idx < 3:
            strides = (1, 1)
        elif 500 < margin < 5000 and block_idx < 2:
            strides = (1, 1)
        else:
            strides = (1, 2)
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, hidden_filter=hidden_filter, hidden_dilation=hidden_dilation,
                             strides=strides, dropout_rate=dropout_rate, weight_decay=weight_decay, block_idx=block_idx)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = BatchNormalization(axis=concat_axis, epsilon=1.001e-5, name='bn')(x)

    rnnout = TimeDistributed(Bidirectional(LSTM(128, activation='linear', return_sequences=True)))(x)
    rnnout_gate = TimeDistributed(Bidirectional(LSTM(128, activation='sigmoid', return_sequences=True)))(x)

    x = Multiply()([rnnout, rnnout_gate])

    x = Conv2D(num_classes, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)

    return out


def DeepSNP(input_tensor, conv_architecture, first_filter, hidden_filter, first_dilation, hidden_dilation,
            use_input_attention, use_output_attention, margin, dropout_rate, num_classes, depth=16, nb_dense_block=6,
            growth_rate=12, nb_filter=16, weight_decay=1E-4):
    """
    Build the DeepSNP V1 model
    :param input_tensor:            input tensor
    :param conv_architecture:       convolutional architecture to use
    :param first_filter:            number of first filters
    :param hidden_filter:           number of hidden filters
    :param first_dilation:          number of first dilations
    :param hidden_dilation:         number of hidden dilations
    :param use_input_attention:     use input attention flag
    :param use_output_attention:    use output attention flag
    :param margin:                  data margin
    :param dropout_rate:            dropout rate
    :param num_classes:             number of classes
    :param depth:                   number or layers
    :param nb_dense_block:          number of dense blocks to add to end
    :param growth_rate:             number of filters to add
    :param nb_filter:               number of filters
    :param weight_decay:            weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """

    if conv_architecture == 'VGG':
        x = VGGFeats(input_tensor=input_tensor, first_filter=first_filter, hidden_filter=hidden_filter,
                     first_dilation=first_dilation, hidden_dilation=hidden_dilation)
    elif conv_architecture == 'DenseNet':
        x = DenseNetFeats(input_tensor=input_tensor, first_filter=first_filter, hidden_filter=hidden_filter,
                          first_dilation=first_dilation, hidden_dilation=hidden_dilation, margin=margin,
                          nb_filter=nb_filter, weight_decay=weight_decay, growth_rate=growth_rate,
                          dropout_rate=dropout_rate, nb_dense_block=nb_dense_block, depth=depth)
    else:
        raise ValueError('Unexpected conv architecture type %s' % conv_architecture)

    if use_input_attention:
        x_gate = Permute((2, 1, 3), input_shape=tuple(x.shape.as_list()))(x)
        x_gate_sig = TimeDistributed(Dense(int(input_tensor.shape[-1]), activation='sigmoid'),
                                     name='input_attention_layer')(x_gate)
        x_gate = Lambda(attend_in)([x_gate, x_gate_sig])
        x_gate = Permute((2, 1, 3), input_shape=tuple(x_gate.shape.as_list()))(x_gate)
    else:
        x_gate = x

    # Gated BGRU
    rnnout = TimeDistributed(Bidirectional(GRU(128, activation='linear', return_sequences=True)))(x_gate)
    rnnout_gate = TimeDistributed(Bidirectional(GRU(128, activation='sigmoid', return_sequences=True)))(x_gate)

    a5 = Multiply()([rnnout, rnnout_gate])

    a6 = Permute((2, 1, 3), input_shape=tuple(a5.shape.as_list()))(a5)
    a7 = Reshape((int(a6.shape[1]), int(a6.shape[2]) * int(a6.shape[3])))(a6)

    sig = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='attention_layer')(a7)
    soft = TimeDistributed(Dense(num_classes, activation='softmax'), name='localization_layer')(a7)

    if use_output_attention:
        out = Lambda(attend_out, output_shape=(num_classes,), name='output_layer')([soft, sig])
    else:
        out = Lambda(avg, output_shape=(num_classes,), name='output_layer')(soft)

    return out


def DeepSNP_V2(input_tensor, first_filter, hidden_filter, first_dilation, hidden_dilation, use_input_attention,
               use_output_attention, margin, dropout_rate, num_classes, depth=16, nb_dense_block=6, growth_rate=12,
               nb_filter=16, weight_decay=1E-4):
    """
    Build the DeepSNP V2 model
    :param input_tensor:            input tensor
    :param first_filter:            number of first filters
    :param hidden_filter:           number of hidden filters
    :param first_dilation:          number of first dilations
    :param hidden_dilation:         number of hidden dilations
    :param use_input_attention:     use input attention flag
    :param use_output_attention:    use output attention flag
    :param margin:                  data margin
    :param dropout_rate:            dropout rate
    :param num_classes:             number of classes
    :param depth:                   number or layers
    :param nb_dense_block:          number of dense blocks to add to end
    :param growth_rate:             number of filters to add
    :param nb_filter:               number of filters
    :param weight_decay:            weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(filters=nb_filter, kernel_size=first_filter,
               strides=first_dilation, padding='valid', name='initial_conv2D',
               kernel_regularizer=l2(weight_decay))(input_tensor)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if margin <= 500 and block_idx < 3:
            strides = (1, 1)
        elif 500 < margin < 5000 and block_idx < 2:
            strides = (1, 1)
        else:
            strides = (1, 2)
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, hidden_filter=hidden_filter, hidden_dilation=hidden_dilation,
                             strides=strides, dropout_rate=dropout_rate, weight_decay=weight_decay, block_idx=block_idx)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    if use_input_attention:
        x_gate = Permute((2, 1, 3), input_shape=tuple(x.shape.as_list()))(x)
        x_gate_sig = TimeDistributed(Dense(int(input_tensor.shape[-1]), activation='sigmoid'),
                                     name='input_attention_layer')(x_gate)
        x_gate = Lambda(attend_in)([x_gate, x_gate_sig])
        x_gate = Permute((2, 1, 3), input_shape=tuple(x_gate.shape.as_list()))(x_gate)
    else:
        x_gate = x

    # Gated BGRU
    rnnout = TimeDistributed(Bidirectional(GRU(128, activation='linear', return_sequences=True)))(x_gate)
    rnnout_gate = TimeDistributed(Bidirectional(GRU(128, activation='sigmoid', return_sequences=True)))(x_gate)
    a5 = Multiply()([rnnout, rnnout_gate])

    a6 = Permute((2, 1, 3), input_shape=tuple(a5.shape.as_list()))(a5)
    a7 = Reshape((int(a6.shape[1]), int(a6.shape[2]) * int(a6.shape[3])))(a6)

    sig = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='attention_layer')(a7)
    soft = TimeDistributed(Dense(num_classes, activation='softmax'), name='localization_layer')(a7)

    if use_output_attention:
        out = Lambda(attend_out, output_shape=(num_classes,), name='output_layer')([soft, sig])
    else:
        out = Lambda(avg, output_shape=(num_classes,), name='output_layer')(soft)

    return out


def DenseNetFeats(input_tensor, first_filter, hidden_filter, first_dilation, hidden_dilation, margin, nb_filter,
                  weight_decay, growth_rate, dropout_rate, nb_dense_block=6, depth=16):
    """
    Build the DenseNet Feature model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param first_dilation:  number of first dilations
    :param hidden_dilation: number of hidden dilations
    :param margin:          data margin
    :param dropout_rate:    dropout rate
    :param depth:           number or layers
    :param nb_dense_block:  number of dense blocks to add to end
    :param growth_rate:     number of filters to add
    :param nb_filter:       number of filters
    :param weight_decay:    weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)
    # Initial convolution
    x = Conv2D(filters=nb_filter, kernel_size=first_filter, dilation_rate=first_dilation, padding='valid',
               name='initial_conv2D', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay),
               kernel_initializer="he_uniform")(input_tensor)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if margin <= 500 and block_idx < 3:
            strides = (1, 1)
        elif 500 < margin < 5000 and block_idx < 2:
            strides = (1, 1)
        else:
            strides = (1, 2)
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, hidden_filter=hidden_filter, hidden_dilation=hidden_dilation,
                             dropout_rate=dropout_rate, weight_decay=weight_decay, block_idx=block_idx, strides=strides)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    return x


def DilBLDenseNetFeats(input_tensor, hidden_filter, first_filter, margin, nb_filter, weight_decay, growth_rate,
                       dropout_rate, nb_dense_block=6, depth=16):
    """
    Build the Dilation baseline DenseNet Feature model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param margin:          data margin
    :param dropout_rate:    dropout rate
    :param depth:           number or layers
    :param nb_dense_block:  number of dense blocks to add to end
    :param growth_rate:     number of filters to add
    :param nb_filter:       number of filters
    :param weight_decay:    weight decay
    :return: keras tensor with nb_layers of conv_block appended
    """
    dil_list = [3, 3 ** 2, 3 ** 3, 3 ** 4, 3 ** 5, 3 ** 6, 3 ** 7, 3 ** 8]

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(filters=nb_filter, kernel_size=first_filter, dilation_rate=dil_list[0], padding='valid',
               name='initial_conv2D', kernel_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
               kernel_initializer="he_uniform")(input_tensor)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        if margin <= 500 and block_idx < 3:
            strides = (1, 1)
        elif 500 < margin < 5000 and block_idx < 2:
            strides = (1, 1)
        else:
            strides = (1, 2)
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = dil_transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                 block_idx=block_idx, strides=strides, dilation=dil_list[block_idx + 1])

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    return x


def VGGFeats(input_tensor, first_filter, hidden_filter, first_dilation, hidden_dilation):
    """
    Build the VGG Feature model
    :param input_tensor:    input tensor
    :param first_filter:    number of first filters
    :param hidden_filter:   number of hidden filters
    :param first_dilation:  number of first dilations
    :param hidden_dilation: number of hidden dilations
    :return: keras tensor with nb_layers of conv_block appended
    """
    x = Conv2D(filters=64, kernel_size=first_filter, dilation_rate=first_dilation, padding='valid',
               name='initial_conv2D', kernel_initializer="he_uniform")(input_tensor)
    x = Conv2D(64, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(hidden_filter, strides=first_dilation, name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(hidden_filter, strides=hidden_dilation, name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(hidden_filter, strides=hidden_dilation, name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(hidden_filter, strides=hidden_dilation, name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (1, hidden_filter[1] + 1), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(hidden_filter, strides=hidden_dilation, name='block5_pool')(x)

    return x
