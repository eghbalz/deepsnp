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
import os
import time
import shutil
from network.NetRunner import NetRunner
import numpy as np
from utils.StratifiedImageGenerator import StratifiedImageGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from utils.utils import CustomPrintLR, lr_decay_callback
from utils.TrainValTensorBoard import TrainValTensorBoard


class TrainRunner(NetRunner):
    """
    DeepSNP TrainRunner class. Base class for training.
    """
    def __init__(self, args=None, experiment_id=None):
        """
        Initialize TrainRunner
        :param args:            configuration parameters
        :param experiment_id:   string identifying the experiment to run
        """
        super().__init__(args, experiment_id)

    def start_training(self):
        """
        Start training
        :return:
        """

        print('... starting training ...')
        n_fold = 0
        for train_index, valid_index in self.kfold.split(self.kf_X):

            for margin in self.margins:

                model_id = '{}{}_fold{}_att{}_{}_margin{}'.format(self.model_name, self.conv_architecture, n_fold,
                                                                  self.use_input_attention, self.use_output_attention,
                                                                  margin)

                model_dir = os.path.join(self.model_dir, model_id)
                if not os.path.exists(model_dir):
                    print('... creating model folder ({}) ...'.format(model_dir))
                    os.makedirs(model_dir)
                log_dir = os.path.join(self.log_dir, '{}_date_{}'.format(os.path.basename(model_dir),
                                                                         time.strftime("%d_%m_%Y", time.localtime(
                                                                             time.time()))))
                # remove existing folder
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)
                if not os.path.exists(log_dir):
                    print('... creating log folder ({}) ...'.format(log_dir))
                    os.makedirs(log_dir)

                # load data
                x_train, y_train, num_classes, classes = self.load_data(train_index, margin=margin, mode_id='(train) ')
                x_valid, y_valid, _, _ = self.load_data(valid_index, margin=margin, mode_id='(valid) ')

                print('Train y: {}, valid y: {}'.format(np.sum(y_train, axis=0), np.sum(y_valid, axis=0)))

                self.input_shape = x_train.shape[1:]
                datagen = StratifiedImageGenerator(featurewise_center=True,
                                                   featurewise_std_normalization=True)
                datagen.fit(x_train)
                train_generator = datagen.stratified_iterator(x_train, y_train, batch_size=self.batch_size,
                                                              classes=classes)

                # load model
                model, loc_model = self.select_model(margin=margin, num_classes=num_classes)

                # define callbacks
                checkpoint = ModelCheckpoint(model_dir + "-ep{epoch:04d}-val_loss{val_loss:.4f}.h5",
                                             monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, period=self.save_interval)
                if self.optimizer is not 'sgd':
                    earlystop = EarlyStopping(monitor='val_loss', min_delta=self.early_stopping_min_delta,
                                              patience=self.early_stopping_patience, verbose=1)

                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.ref_factor,
                                                  patience=self.ref_patience, min_lr=self.ref_min_lr, verbose=1)
                else:
                    # reduce_lr = None
                    print_call = CustomPrintLR()
                    reduce_lr = lr_decay_callback(init_epoch=self.epochs, learn_rate=self.lr,
                                                  reduce_list=self.lr_reduce_rates)

                tensorboard = TrainValTensorBoard(log_dir=log_dir, write_graph=False)

                if self.optimizer is not 'sgd':
                    callbacks_list = [checkpoint, earlystop, reduce_lr, tensorboard]
                else:
                    callbacks_list = [checkpoint, print_call, reduce_lr, tensorboard]

                # start training
                model.fit_generator(train_generator, validation_data=(x_valid, y_valid),
                                    steps_per_epoch=len(x_train) // self.batch_size, epochs=self.epochs, verbose=1,
                                    workers=4, callbacks=callbacks_list)

                # saving the final model is mandatory
                model.save(os.path.join(model_dir + "_fully_trained_model.h5"))

            n_fold += 1
