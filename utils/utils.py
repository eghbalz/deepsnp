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
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler


def lr_decay_callback(init_epoch, learn_rate, reduce_list, decay=0.1):
    """
    Custom learning rate decay callback
    :param init_epoch:  initial epoch
    :param learn_rate:  learning rate
    :param reduce_list: list of them point when to reduce learning rate
    :param decay:       learning rate decay
    :return:
    """
    def step_decay(epoch):
        return learn_rate * decay if epoch in [int(init_epoch / i) for i in reduce_list] else learn_rate

    return LearningRateScheduler(step_decay)


class CustomPrintLR(keras.callbacks.Callback):
    """
    Custom print learning rate callback
    """
    def on_epoch_begin(self, epoch, logs=None):
        print('Begin epoch {:d} learning rate {:.4f}'.format(epoch, K.eval(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        print('End epoch {:d} learning rate {:.4f}'.format(epoch, K.eval(self.model.optimizer.lr)))
