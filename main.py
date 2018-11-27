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
import numpy as np
import argparse

from network.InferenceRunner import InferenceRunner
from network.TrainRunner import TrainRunner
from network.RawDataRunner import RawDataRunner
from network.RawCopyRunner import RawCopyRunner

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True      # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

np.random.seed(666)


def run(experiment_id, mode):
    if mode == 'training':
        training = TrainRunner(experiment_id=experiment_id)
        training.start_training()
    elif mode == 'inference':
        inference = InferenceRunner(experiment_id=experiment_id)
        inference.start_testing()
    elif mode == 'data_from_raw':
        data_gen = RawDataRunner(experiment_id='process_raw_data')
        data_gen.process_raw_data()
    elif mode == 'eval_rawcopy':
        rawcopy = RawCopyRunner(experiment_id='rawcopy')
        rawcopy.evaluate()
    else:
        raise ValueError('Only \'training\', \'inference\', \'data_from_raw\' or \'eval_rawcopy\' mode are allowed')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepSNP: An End-to-end Deep Neural Network with Attention-based '
                                                 'Localization for Breakpoint Detection in SNP Array Genomic Data')

    parser.add_argument("-e", "--exp_id", default='DeepSNP_V1_finalAtt', type=str, help="Experiment ID")
    parser.add_argument("-m", "--mode", default='inference', type=str, help="DeepSNP run mode: training, inference,"
                                                                            "data_from_raw, 'eval_rawcopy")

    args = parser.parse_args()

    run(args.exp_id, args.mode)
