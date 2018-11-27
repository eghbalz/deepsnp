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
import h5py
import glob
import numpy as np
import importlib
from sklearn.model_selection import KFold

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.optimizers import adam, SGD

from utils import models
from utils.metrics import f1_score, precision_score, recall_score
from utils.prepare_data import get_patient_ids


class NetRunner:
    """
    DeepSNP NetRunner class. Base class for training and inference.
    """

    def __init__(self, args=None, experiment_id=None):
        """
        Initialize NetRunner
        :param args:            configuration parameters
        :param experiment_id:   string identifying the experiment to run
        """
        self.input_shape = None
        self.model = None
        self.loc_model = None
        self._parse_config(args, experiment_id)

    def _parse_config(self, args, experiment_id):
        """
        Read parameters from config files
        :param args:            configuration parameters
        :param experiment_id:   string identifying the experiment to run
        :return:
        """

        if not args:
            if experiment_id:
                try:
                    config = importlib.import_module('configs.config_' + experiment_id)
                    args = config.load_config()
                except ModuleNotFoundError:
                    print('Config for {} does not exist! Falling back to standard config!'.format(experiment_id))
                    config = importlib.import_module('configs.config')
                    args = config.ConfigFlags().return_flags()
                    args.model_name = experiment_id

            else:
                config = importlib.import_module('configs.config')
                args = config.ConfigFlags().return_flags()

        # Directories
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            print('... creating save directory ({})'.format(self.save_dir))
            os.mkdir(self.save_dir)
        self.model_dir = args.model_dir
        if not os.path.exists(self.model_dir):
            print('... creating model directory ({})'.format(self.model_dir))
            os.mkdir(self.model_dir)
        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            print('... creating logging directory ({})'.format(self.log_dir))
            os.mkdir(self.log_dir)

        self.feat_dir = os.path.join(self.data_dir, 'features_{}'.format(self.data_dir.split('Genomicdata_')[-1]))

        # Model to train
        self.model_name = args.model_name

        # Data generation parameters
        if args.margin < 100:
            self.margins = np.array([10000, 5000, 2500, 1250])
        else:
            self.margins = np.array([args.margin])
        self.use_jitter = args.use_jitter

        # Training parameters
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_splits = args.n_splits

        # todo implement continue train
        # self.fold = args.fold
        # self.continue_train = args.continue_train

        self.early_stopping_min_delta = args.early_stopping_min_delta
        if not args.early_stopping_patience:
            self.early_stopping_patience = int(self.epochs / 2)
        else:
            self.early_stopping_patience = args.early_stopping_patience
        self.ref_factor = args.ref_factor
        self.ref_min_lr = args.ref_min_lr
        if not args.ref_patience:
            self.ref_patience = int(self.epochs / 6)
        else:
            self.ref_patience = args.ref_patience
        self.optimizer = args.optimizer
        if self.optimizer == 'sgd':
            # adapt learn rate according to batch size
            self.lr = self.lr * ((self.batch_size / (self.batch_size / 2)) ** 0.5)
        self.amsgrad = args.amsgrad
        self.loss = args.loss
        self.dense_clf = args.dense_clf
        self.save_interval = args.save_interval

        # Architecture specific parameters
        self.first_dilation = (1, args.first_dilation)
        self.hidden_dilation = (1, args.hidden_dilation)
        self.first_filter = (1, args.first_filter)
        self.hidden_filter = (1, args.hidden_filter)
        self.dropout_rate = args.dropout_rate
        self.conv_architecture = args.conv_architecture
        self.use_input_attention = args.use_input_attention
        self.use_output_attention = args.use_output_attention

        # Evaluation parameters
        self.eval_num_classes = args.eval_num_classes
        self.pred_thresh = args.pred_thresh
        self.gen_loc_output = args.gen_loc_output
        self.plot_loc_results = args.plot_loc_results

        # Raw data processing parameters
        self.hop_modifiers = args.hop_modifiers
        self.default_data_gen_margin = args.default_data_gen_margin
        self.bp_delta_thresh = args.bp_delta_thresh
        self.plot_data_gen = args.plot_data_gen

        # GPU config
        self.per_process_gpu_mem_frac = args.per_process_gpu_mem_frac
        self.allow_gpu_growth = args.allow_gpu_growth
        self._set_gpu_config()

        # general config
        self.random_seed = args.random_seed

        # initialize data
        self._initialize_data()

    def _set_gpu_config(self):
        """
        Set GPU options
        :return:
        """
        print('... setting GPU config ...')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_mem_frac,
                                    allow_growth=self.allow_gpu_growth)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        set_session(tf.Session(config=tf_config))
        print('... done ...')

    def _initialize_data(self):
        """
        Initialize data to compute cross validation information, patient ids and patient count
        :return:
        """

        # load pat id data
        h5f = h5py.File(os.path.join(self.feat_dir, 'patient_ids.h5'), 'r')
        pat_id_refs = h5f['pat_ids'][:]
        h5f.close()

        pat_ids = []
        for bn in pat_id_refs:
            _ind = bn.find('_') + 1
            pat_ids.append(bn[_ind:_ind + 3])

        upis, pis, count = np.unique(pat_ids, return_index=True, return_counts=True)
        pats = [pat_ids[pi] for pi in sorted(pis)]
        n_pats = len(pats)

        self.kfold = KFold(n_splits=self.n_splits, random_state=self.random_seed, shuffle=False)
        self.kf_X = np.arange(0, n_pats)

        self.pat_id_refs = pat_id_refs
        self.pat_ids = dict(zip(pats, self.kf_X))
        self.pat_count = dict(zip(upis, count))

    def load_data(self, pat_ind, margin, mode_id=''):
        """
        Load data and create split based on patient ids
        :param pat_ind: patient indices to load
        :param margin:  load data that was created with respective margin
        :param mode_id: mode identifier (train, test, etc)
        :return:
        """

        print('... loading data {}...'.format(mode_id))
        # load pat id data
        pat_inds = get_patient_ids(pat_ind, self.pat_ids, self.pat_count, self.data_dir)

        raw_feat_file = os.path.join(self.feat_dir, 'raw_features_margin{}'.format(margin))
        raw_label_file = os.path.join(self.feat_dir, 'raw_labels_margin{}'.format(margin))
        if self.use_jitter:
            jitt_str = '_jitter.h5'
        else:
            jitt_str = '.h5'

        raw_feat_file += jitt_str
        raw_label_file += jitt_str

        h5f = h5py.File(raw_feat_file, 'r')
        x_raw = np.array([i[:] for i in h5f.values()])[pat_inds]
        h5f.close()

        h5f = h5py.File(raw_label_file, 'r')
        y_raw = np.array([i[:] for i in h5f.values()])[pat_inds]
        h5f.close()

        x, y = [], []
        for y_ind, y_r in enumerate(y_raw):
            x_t = x_raw[y_ind]

            for l_ind, l in enumerate(y_r):
                x.append(x_t[l_ind])
                # only one label, no change
                if len(l) == 3:
                    y.append(0)
                else:
                    y.append(1)

            assert (len(y) == len(x))

        classes = np.unique(y)
        num_classes = len(classes)

        y = to_categorical(y, num_classes).astype(np.int32)

        x = np.array(x)
        x = x[:, :, :, None]

        print('... done ...')

        return x, y, num_classes, classes

    def load_test_rc_data(self, test_patient_ind, margin):
        """
        Load Rawcopy data for inference
        :param test_patient_ind: patient ID to load
        :param margin:          margin to use
        :return:
        """
        raw_label_file = os.path.join(self.data_dir, 'test_raw_labels_margin{}'.format(margin))
        raw_rc_preds_file = os.path.join(self.data_dir, 'test_raw_rc_preds_margin{}'.format(margin))
        if self.use_jitter:
            jitt_str = '_jitter.h5'
        else:
            jitt_str = '.h5'

        raw_label_file += jitt_str
        raw_rc_preds_file += jitt_str

        # load pat id data
        pat_inds = get_patient_ids(test_patient_ind, self.pat_ids, self.pat_count, self.data_dir)

        h5f = h5py.File(raw_label_file, 'r')
        y_train_raw = np.array([i[:] for i in h5f.values()])[pat_inds]

        h5f = h5py.File(raw_rc_preds_file, 'r')
        preds_train_raw = np.array([i[:] for i in h5f.values()])[pat_inds]

        y_train, preds_train = [], []
        for y_ind, y_raw in enumerate(y_train_raw):
            p_t = preds_train_raw[y_ind]
            for l_ind, l in enumerate(y_raw):

                if len(p_t[l_ind]) == 3:
                    preds_train.append(0)
                else:
                    preds_train.append(1)
                if len(l) == 3:
                    y_train.append(0)
                else:
                    y_train.append(1)

            assert (len(y_train) == len(preds_train))

        num_classes = len(np.unique(y_train))

        y_train = to_categorical(y_train, num_classes).astype(np.int32)

        preds_train = to_categorical(preds_train, num_classes).astype(np.int32)

        return y_train, preds_train

    def select_model(self, margin, num_classes):
        """
        Selcet a neural network model, load and compile it
        :param margin:      data margin
        :param num_classes: number of classes
        :return:
        """
        input_tensor = Input(shape=self.input_shape, name='image_input')

        if self.model_name == 'BLVGG':
            print('Model BLVGG is loaded')
            predictions = models.BLVGG(input_tensor=input_tensor, first_filter=self.first_filter,
                                       hidden_filter=self.hidden_filter, first_dilation=self.first_dilation,
                                       num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            loc_model = model

        elif self.model_name == 'BLDilDenseNet':
            print('Model BLDilDenseNet (margin: {}) is loaded'.foramt(margin))
            predictions = models.BLDilDenseNet(input_tensor=input_tensor, first_filter=self.first_filter,
                                               hidden_filter=self.hidden_filter, margin=margin,
                                               dropout_rate=self.dropout_rate, num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            loc_model = model

        elif self.model_name == 'BLLSTMDenseNet':
            print('Model BLLSTMDenseNet (margin: {}) is loaded'.foramt(margin))
            predictions = models.BLLSTMDenseNet(input_tensor=input_tensor, first_filter=self.first_filter,
                                                hidden_filter=self.hidden_filter, first_dilation=self.first_dilation,
                                                hidden_dilation=self.hidden_dilation, margin=margin,
                                                dropout_rate=self.dropout_rate, num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            loc_model = model

        elif self.model_name == 'BLDenseNet':
            print('Model BLDenseNet (margin: {}) is loaded'.foramt(margin))
            predictions = models.BLDenseNet(input_tensor=input_tensor, first_filter=self.first_filter,
                                            hidden_filter=self.hidden_filter, first_dilation=self.first_dilation,
                                            hidden_dilation=self.hidden_dilation, margin=margin,
                                            dropout_rate=self.dropout_rate, num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            loc_model = model

        elif self.model_name == 'DeepSNP_V1':
            print('Model DeepSNP V1 (input att: {}, output att: {}, margin: {}) '
                  'is loaded'.format(self.use_input_attention, self.use_output_attention, margin))
            predictions = models.DeepSNP(input_tensor=input_tensor, conv_architecture=self.conv_architecture,
                                         first_filter=self.first_filter, hidden_filter=self.hidden_filter,
                                         first_dilation=self.first_dilation, hidden_dilation=self.hidden_dilation,
                                         use_input_attention=self.use_input_attention,
                                         use_output_attention=self.use_output_attention, margin=margin,
                                         dropout_rate=self.dropout_rate, num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            layer_name = 'localization_layer'
            loc_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        elif self.model_name == 'DeepSNP_V2':
            print('Model DeepSNP V2 (input att: {}, output att: {}, margin: {}) '
                  'is loaded'.format(self.use_input_attention, self.use_output_attention, margin))
            predictions = models.DeepSNP_V2(input_tensor=input_tensor, first_filter=self.first_filter,
                                            hidden_filter=self.hidden_filter, first_dilation=self.first_dilation,
                                            hidden_dilation=self.hidden_dilation, margin=margin,
                                            dropout_rate=self.dropout_rate, num_classes=num_classes)
            model = Model(inputs=input_tensor, outputs=predictions)
            layer_name = 'localization_layer'
            loc_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        else:
            raise ValueError('Unexpected model {}'.format(self.model))

        # Setup Optimizer
        if self.optimizer is 'sgd':
            opt = SGD(lr=self.lr, momentum=0.9, clipvalue=5, nesterov=True)
        elif self.optimizer is 'adam':
            opt = adam(lr=self.lr, decay=0.1, amsgrad=self.amsgrad)
        else:
            raise ValueError('Unexpected optimizer {}'.format(self.optimizer))

        # Setup Metrics
        if self.loss == 'binary_crossentropy':
            metrics = ['binary_accuracy', f1_score, precision_score, recall_score]

        else:
            metrics = ['categorical_accuracy', f1_score, precision_score, recall_score]

        # Compile Model
        model.compile(optimizer=opt, loss=self.loss, metrics=metrics)

        model.summary()

        return model, loc_model

    def load_model(self, model_id, margin, num_classes):
        """
        Load a pre-trained model (weights)
        :param model_id:    model identifier string
        :param margin:      data margin, the model was trained on
        :param num_classes: number of classes
        :return:
        """
        # load model
        input_shape = (2, margin * 4, 1)
        # get last model weights file
        end_id = model_id.find('margin{}'.format(margin))
        model_files = glob.glob(
            os.path.join(self.model_dir, model_id[:end_id + len('margin{}'.format(margin))] + '_*'))
        if model_files:
            model, loc_model = self.select_model(margin, num_classes)

            model_losses = [float(mf.split('-')[-1][8:-3]) if 'val_loss' in mf.split('-')[-1] else np.inf for mf
                            in model_files]
            model_file = model_files[np.argmin(model_losses)]
            model.load_weights(model_file)
            model.summary()
        else:
            print('Model does not exist! Skipping!')
            model = None
            loc_model = None

        return model, loc_model
