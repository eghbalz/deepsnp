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


class ConfigFlags:
    def __init__(self):
        """
        DeepSNP parameter configuration
        """
        self.data_dir =     r'\data'        # Path and name of data directory
        self.save_dir =     r'\results'     # Path and name of save directory
        self.model_dir =    r'\models'      # Path and name of model directory
        self.log_dir =      r'\logs'        # Path and name of logging directory

        # Model to train
        self.model_name = None  # Name of the model to train. DeepSNP or Baselines

        # Data generation parameters
        self.margin =       1       # Window margin
        self.use_jitter =   True    # Use jittered data

        # Training parameters
        self.epochs =       200     # Number of epochs
        self.lr =           0.001   # Learning rate
        self.batch_size =   25      # Batch size
        self.n_splits =     6       # Number of cross validation splits

        # todo implement continue train
        # self.fold =           -1      # Cross validation fold. For re-runs
        # self.continue_train = False   # Continue training

        self.early_stopping_min_delta = 0.0001                      # Early stopping min loss delta
        self.early_stopping_patience =  None                        # Early stopping patience
        self.ref_factor =               0.5                         # Refinement factor
        self.ref_min_lr =               0.00001                     # Minimum refinement learning rate
        self.ref_patience =             None                        # Refinement patience
        self.optimizer =                'adam'                      # Optimizer used for training
        self.amsgrad =                  True                        # Use AMSGrad for Adam optimizer
        self.loss =                     'categorical_crossentropy'  # training loss
        self.dense_clf =                True                        # todo description
        self.save_interval =            10                          # Intervall to save models/checkpoints during
                                                                    # training in epochs
        self.lr_reduce_rates =          [1 / 2, 2 / 3, 3 / 4]       # Learning rate reduction rates (epochs/rate).

        # Architecture specific parameters
        self.first_dilation =       5           # First dilation
        self.hidden_dilation =      2           # Hidden dilation
        self.first_filter =         10          # First filter
        self.hidden_filter =        5           # Hidden filter
        self.dropout_rate =         0.2         # Dropout rate
        self.conv_architecture =    'DenseNet'  # Convolution architecture to use.
        self.use_input_attention =  False       # Use input attention
        self.use_output_attention = False       # Use output attention

        # Evaluation parameters
        self.eval_num_classes = 2       # Number of classes, needed for evaluation
        self.pred_thresh =      .5      # Threshold for BP prediction
        self.gen_loc_output =   False   # Generate localization unit output
        self.plot_loc_results = False   # Plot localization unit results

        # Raw data processing parameters
        self.hop_modifiers = [0.125, 0.125 / 2, 0.025, 0.0125, 0.006, 0.0025]   # Hop modifiers for margin/window
                                                                                # size generation
        self.default_data_gen_margin =  10000   # Default window margin used for data generation
        self.bp_delta_thresh =          0.001   # Threshold for breakpoints
        self.plot_data_gen =            False   # Plot generated data windows

        # GPU config
        self.per_process_gpu_mem_frac = 0.8     # Per process GPU Memory fraction
        self.allow_gpu_growth =         True    # Allow GPU growth

        # general config
        self.random_seed = 47  # Random seed for cross validation, etc.
