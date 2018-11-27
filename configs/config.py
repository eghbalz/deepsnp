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
import argparse


class ConfigFlags:
    def __init__(self):
        """
        DeepSNP parameter configuration
        """
        parser = argparse.ArgumentParser(description='DeepSNP: An End-to-end Deep Neural Network with Attention-based '
                                                     'Localization for Breakpoint Detection in SNP Array Genomic Data')
        # Directories
        parser.add_argument("-d", "--data_dir", default=r'\data', type=str, help="Path and name of data directory")
        parser.add_argument("-s", "--save_dir", default=r'\results', type=str, help="Path and name of save directory")
        parser.add_argument("-m", "--model_dir", default=r'\models', type=str, help="Path and name of model directory")
        parser.add_argument("-l", "--log_dir", default=r'\logs', type=str, help="Path and name of logging directory")

        # Model to train
        parser.add_argument("-mn", "--model_name", type=str, help="Name of the model to train. DeepSNP or Baselines")

        # Data generation parameters
        parser.add_argument("-wm", "--margin", default=1, type=float, help="Window margin")
        parser.add_argument("-j", "--use_jitter", default=True, type=bool, help="Use jittered data")

        # Training parameters
        parser.add_argument("-e", "--epochs", default=200, type=int, help="Number of epochs")
        parser.add_argument("-lr", default=0.001, type=float, help="Learning rate")
        parser.add_argument("-b", "--batch_size", default=25, type=int, help="Batch size")
        parser.add_argument("-ns", "--n_splits", default=6, type=int, help="Number of cross validation splits")

        # todo implement continue train
        # parser.add_argument("-f", "--fold", default=-1, type=int, help="Cross validation fold. For re-runs.")
        # parser.add_argument("--continue_train", action='store_true', help="Continue training")

        parser.add_argument("--early_stopping_min_delta", default=0.0001, type=float,
                            help="Early stopping min loss delta")
        parser.add_argument("--early_stopping_patience", type=float, help="Early stopping patience")
        parser.add_argument("--ref_factor", default=0.5, type=float, help="Refinement factor")
        parser.add_argument("--ref_min_lr", default=0.00001, type=float, help="Minimum refinement learning rate")
        parser.add_argument("--ref_patience", type=float, help="Refinement patience")
        parser.add_argument("--optimizer", default='adam', type=str, help="Optimizer used for training")
        parser.add_argument("--amsgrad", default=True, type=bool, help="Use AMSGrad for Adam optimizer")
        parser.add_argument("--loss", default='categorical_crossentropy', type=str, help="training loss")
        parser.add_argument("--dense_clf", default=True, type=bool, help="")  # todo description
        parser.add_argument("--save_interval", default=10, type=int,
                            help="Intervall to save models/checkpoints during training in epochs")
        parser.add_argument("--lr_reduce_rates", default=[1 / 2, 2 / 3, 3 / 4], type=list,
                            help="Learning rate reduction rates (epochs/rate).")

        # Architecture specific parameters
        parser.add_argument("-fd", "--first_dilation", default=5, type=int, help="First dilation")
        parser.add_argument("-hd", "--hidden_dilation", default=2, type=int, help="Hidden dilation")
        parser.add_argument("-ff", "--first_filter", default=10, type=int, help="First filter")
        parser.add_argument("-hf", "--hidden_filter", default=5, type=int, help="Hidden filter")
        parser.add_argument("-dr", "--dropout_rate", default=0.2, type=float, help="Dropout rate")
        parser.add_argument("-cf", "--conv_architecture", default='DenseNet', type=str,
                            help="Convolution architecture to use.")
        parser.add_argument("-ia", "--use_input_attention", action='store_true', help="Use input attention")
        parser.add_argument("-oa", "--use_output_attention", action='store_true', help="Use output attention")

        # Evaluation parameters
        parser.add_argument("--eval_num_classes", default=2, type=int, help="Number of classes, needed for evaluation")
        parser.add_argument("--pred_thresh", default=.5, type=float, help="Threshold for BP prediction")
        parser.add_argument("--gen_loc_output", action='store_true', help="Generate localization unit output")
        parser.add_argument("--plot_loc_results", action='store_true', help="Plot localization unit results")

        # Raw data processing parameters
        parser.add_argument("--hop_modifiers", default=[0.125, 0.125 / 2, 0.025, 0.0125, 0.006, 0.0025], type=list,
                            help="Hop modifiers for margin/window size generation")
        parser.add_argument("--default_data_gen_margin", default=10000, type=float,
                            help="Default window margin used for data generation")
        parser.add_argument("--bp_delta_thresh", default=0.001, type=float, help="Threshold for breakpoints")
        parser.add_argument("--plot_data_gen", action='store_true', help="Plot generated data windows")

        # GPU config
        parser.add_argument("--per_process_gpu_mem_frac", default=0.8, type=float,
                            help="Per process GPU Memory fraction")
        parser.add_argument("--allow_gpu_growth", action='store_true', help="Allow GPU growth")

        # general config
        parser.add_argument("--random_seed", default=47, type=int, help="Random seed for cross validation, etc.")

        self.args = parser.parse_args()

    def return_flags(self):
        """
        Return all parameters
        :return:
        """
        return self.args
