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
import glob
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from network.NetRunner import NetRunner
from utils.prepare_data import feature_extraction, feature_extraction_jitter, plot_lrr_baf


class RawDataRunner(NetRunner):
    def __init__(self, args=None, experiment_id=None):
        super().__init__(args, experiment_id)

    def process_raw_data(self):
        """
        Process raw SNP data provided in .dat files
        :return:
        """

        print('... starting processing raw data ...')

        feat_path = os.path.join(self.data_dir, 'features_' + os.path.basename(self.data_dir))
        if not os.path.exists(feat_path):
            os.mkdir(feat_path)

        # use jittered data
        if self.use_jitter:
            jitter_str = '_jitter.h5'
        else:
            jitter_str = '.h5'

        pat_id_file_path = r'patient_ids.h5'
        default_feat_lab_file_path = r'raw_features_margin{}' + jitter_str
        default_lab_file_path = r'raw_labels_margin{}' + jitter_str
        default_bp_file_path = r'raw_breakpoints_margin{}' + jitter_str
        default_rc_preds_file_path = r'raw_rc_preds_margin{}' + jitter_str
        default_test_feat_lab_file_path = r'test_raw_features_margin{}' + jitter_str
        default_test_lab_file_path = r'test_raw_labels_margin{}' + jitter_str
        default_test_bp_file_path = r'test_raw_breakpoints_margin{}' + jitter_str
        default_test_rc_preds_file_path = r'test_raw_rc_preds_margin{}' + jitter_str

        data_files = glob.glob(os.path.join(self.data_dir, '*.dat'))

        # number of random negative window indices to pick
        n_negative = 4

        # save patient IDs
        pat_ids = []
        for f in data_files:
            pat_ids.append(os.path.basename(f).split('.')[0])

        h5f = h5py.File(os.path.join(feat_path, pat_id_file_path), 'w')
        dt = h5py.special_dtype(vlen=str)
        h5f.create_dataset('pat_ids', data=np.array(pat_ids, dtype='S'), dtype=dt)
        h5f.close()

        file_error = []

        # loop over hop modifiers to generate data margins
        for hm in tqdm(self.hop_modifiers):
            margin = int(self.default_data_gen_margin * hm)

            window_size = margin * 4

            # initialize margin specific lists
            feature_list, test_feat_list = [], []
            label_list, test_label_list, test_strong_label_list = [], [], []
            rawcopy_pred_list, test_rawcopy_preds_list = [], []
            bp_index_list, test_bp_index_list = [], []

            # loop over patients' genomic data samples
            for data_ind, f in enumerate(tqdm(data_files, leave=False)):

                # initialize patient specific data lists
                features, test_feat = [], []
                feat_labels, test_lab, test_strong_lab = [], [], []
                rawcopy_preds, test_rawcopy_preds = [], []
                bp_indices, test_bp_ind = [], []

                # prepare save paths
                feat_lab_file_path = os.path.join(feat_path, default_feat_lab_file_path.format(margin))
                lab_file_path = os.path.join(feat_path, default_lab_file_path.format(margin))
                bp_file_path = os.path.join(feat_path, default_bp_file_path.format(margin))
                rc_preds_file_path = os.path.join(feat_path, default_rc_preds_file_path.format(margin))
                test_feat_lab_file_path = os.path.join(feat_path, default_test_feat_lab_file_path.format(margin))
                test_lab_file_path = os.path.join(feat_path, default_test_lab_file_path.format(margin))
                test_strong_lab_file_path = os.path.join(feat_path, 'test_raw_labels.h5')
                test_bp_file_path = os.path.join(feat_path, default_test_bp_file_path.format(margin))
                test_rc_preds_file_path = os.path.join(feat_path, default_test_rc_preds_file_path.format(margin))

                data = pd.read_csv(f, sep='\t', header=0)

                data_type_names = data.keys()

                # extract data from data frame
                # position = np.array(data._series['Pos'])
                lrr_c = np.array(data._series['LRR'])
                baf_c = np.array(data._series['BAF'])
                labels = np.array(data._series['tCN'])  # tCN
                rawcopy_predictions = np.array(data._series['cCN'])  # cCN
                # gc = np.array(data._series['GC'])  # GC
                # gc_map = np.array(data._series['MAP'])  # MAP
                seq_length = len(labels)

                # get breakpoint indices based on delta (change in signal)
                delta = np.abs(librosa.feature.delta(labels, width=3))
                break_point_index_raw = np.where(delta > self.bp_delta_thresh)[0]

                # fix successive breakpoints - only use one of the tuple
                break_point_index = []
                for ind in range(len(break_point_index_raw) - 1):
                    if break_point_index_raw[ind] + 1 == break_point_index_raw[ind + 1]:
                        break_point_index.append(break_point_index_raw[ind])

                if self.plot_data_gen:
                    plt.figure()
                    plt.subplot(2, 1, 1)
                    plt.plot(labels, '.', markersize=2)
                    plt.subplot(2, 1, 2)
                    plt.plot(labels, '.', markersize=2)
                    for c in break_point_index:
                        plt.plot([c] * 6, range(0, 6))
                    plt.close()

                # check to the left of first break point -> add negative examples
                if break_point_index[0] > window_size:
                    try:
                        random_point = np.random.choice(range(window_size, break_point_index[0] - window_size),
                                                        n_negative, replace=False)
                        for rp in random_point:
                            feat_neg, rawcop_pred_neg, label_neg, ix_neg = feature_extraction(rp, labels, baf_c, lrr_c,
                                                                                              rawcopy_predictions,
                                                                                              data.shape, margin=margin)

                            features.append(feat_neg)
                            feat_labels.append(label_neg)
                            rawcopy_preds.append(rawcop_pred_neg)
                            bp_indices.append([])
                            if self.plot_data_gen:
                                plot_lrr_baf(data, ix_neg, data_type_names, feat_neg,
                                             'first negative BAF - LRR, label: {}'.format(label_neg))
                    except ValueError:
                        print('... skipping negative example ...')

                # run through all breakpoints
                for ind in trange(len(break_point_index) - 1, leave=False):

                    b_ind = break_point_index[ind]
                    next_b_ind = break_point_index[ind + 1]
                    # add negative examples
                    if (next_b_ind - window_size) - (b_ind + window_size) >= window_size:
                        try:
                            random_point = np.random.choice(range(b_ind + window_size, next_b_ind - window_size),
                                                            n_negative, replace=False)

                            for rp in random_point:
                                feat_neg, rawcop_pred_neg, label_neg, ix_neg = feature_extraction(rp, labels, baf_c,
                                                                                                  lrr_c,
                                                                                                  rawcopy_predictions,
                                                                                                  data.shape,
                                                                                                  margin=margin)
                                features.append(feat_neg)
                                feat_labels.append(label_neg)
                                rawcopy_preds.append(rawcop_pred_neg)
                                bp_indices.append([])
                                if self.plot_data_gen:
                                    plot_lrr_baf(data, ix_neg, data_type_names, feat_neg,
                                                 'negative BAF - LRR, label: {}'.format(label_neg))
                        except ValueError:
                            print('... skipping negative example ...')

                    # add positive example
                    if self.use_jitter:
                        feat_pos, rawcop_pred_pos, label_pos = feature_extraction_jitter(b_ind, labels, baf_c, lrr_c,
                                                                                         rawcopy_predictions,
                                                                                         data.shape, margin=margin)
                        if feat_pos is None:
                            file_error.append(f)
                            continue
                    else:
                        feat_pos, rawcop_pred_pos, label_pos, ix_pos = feature_extraction(b_ind, labels, baf_c, lrr_c,
                                                                                          rawcopy_predictions,
                                                                                          data.shape, margin=margin)
                    features.append(feat_pos)
                    feat_labels.append(label_pos)
                    rawcopy_preds.append(rawcop_pred_pos)
                    bp_indices.append([b_ind])

                    if self.plot_data_gen:
                        plot_lrr_baf(data, ix_pos, data_type_names, feat_pos,
                                     'positive BAF - LRR, label: ' + str(label_pos))

                # check to the right of last break point - negative examples
                if seq_length - break_point_index[-1] >= window_size:
                    try:
                        random_point = np.random.choice(range(break_point_index[-1] + window_size,
                                                              seq_length - window_size), n_negative, replace=False)
                        for rp in random_point:
                            feat_neg, rawcop_pred_neg, label_neg, ix_neg = feature_extraction(rp, labels, baf_c, lrr_c,
                                                                                              rawcopy_predictions,
                                                                                              data.shape,
                                                                                              margin=margin)
                            features.append(feat_neg)
                            feat_labels.append(label_neg)
                            rawcopy_preds.append(rawcop_pred_neg)
                            bp_indices.append([])
                            if self.plot_data_gen:
                                plot_lrr_baf(data, ix_neg, data_type_names, feat_neg,
                                             'last negative BAF - LRR, label: {}'.format(label_neg))
                    except ValueError:
                        print('... skipping negative example ...')

                assert (len(features) == len(feat_labels))
                feature_list.append(features)
                label_list.append(feat_labels)
                rawcopy_pred_list.append(rawcopy_preds)
                bp_index_list.append(bp_indices)

                # generate data for testing by windowing over full sequence and recording features,
                # labels and indices of bps
                for fs in range(int(np.ceil(len(labels) / window_size))):
                    st = fs * window_size
                    ed = (fs + 1) * window_size
                    if ed > len(labels):
                        ed = len(labels)
                    f = np.vstack((baf_c[st:ed], lrr_c[st:ed]))
                    # pad data if smaller then 4*margin (window_size)
                    if f.shape[1] < window_size:
                        f = np.pad(f, ((0, 0), (0, window_size - f.shape[1])), 'constant', constant_values=-2)

                    assert f.shape[1] == window_size
                    test_feat.append(f)
                    test_lab.append(np.unique(labels[st:ed]).tolist())
                    test_bp_ind.append(np.where(delta[st:ed] > self.bp_delta_thresh)[0])
                    test_rawcopy_preds.append(np.unique(rawcopy_predictions[st:ed]).tolist())

                test_feat_list.append(test_feat)
                test_label_list.append(test_lab)
                test_strong_label_list.append(np.array(break_point_index))
                test_bp_index_list.append(test_bp_ind)
                test_rawcopy_preds_list.append(test_rawcopy_preds)

            # Save all processed data
            print('... saving data for margin: {} ...'.format(margin))

            h5f = h5py.File(feat_lab_file_path, 'w')
            for ind, feat in enumerate(feature_list):
                h5f.create_dataset('feat_data_' + str(ind), data=feat)
            h5f.close()

            h5f = h5py.File(lab_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(label_list):
                h5f.create_dataset('labels_data_' + str(ind), data=lab, dtype=dt)
            # h5f.create_dataset('labels', data=label_list, dtype=dt)
            h5f.close()

            h5f = h5py.File(bp_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(bp_index_list):
                h5f.create_dataset('breakpoint_data_' + str(ind), data=lab, dtype=dt)
            # h5f.create_dataset('labels', data=label_list, dtype=dt)
            h5f.close()

            # save test data
            h5f = h5py.File(test_feat_lab_file_path, 'w')
            for ind, feat in enumerate(test_feat_list):
                h5f.create_dataset('test_feat_data_' + str(ind), data=feat)
            h5f.close()
            h5f = h5py.File(test_lab_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(test_label_list):
                h5f.create_dataset('test_labels_data_' + str(ind), data=lab, dtype=dt)
            # h5f.create_dataset('labels', data=label_list, dtype=dt)
            h5f.close()

            h5f = h5py.File(test_strong_lab_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset('test_strong_labels_data', data=test_strong_label_list, dtype=dt)
            h5f.close()

            h5f = h5py.File(test_bp_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(test_bp_index_list):
                h5f.create_dataset('test_breakpoint_data_' + str(ind), data=lab, dtype=dt)
            # h5f.create_dataset('labels', data=label_list, dtype=dt)
            h5f.close()

            h5f = h5py.File(rc_preds_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(rawcopy_pred_list):
                h5f.create_dataset('rc_preds_' + str(ind), data=lab, dtype=dt)
            h5f.close()

            h5f = h5py.File(test_rc_preds_file_path, 'w')
            dt = h5py.special_dtype(vlen=str)
            for ind, lab in enumerate(test_rawcopy_preds_list):
                h5f.create_dataset('test_rc_preds_' + str(ind), data=lab, dtype=dt)
            h5f.close()

            # save file error
            np.save(os.path.join(feat_path, 'file_error.npy'), file_error)
            print('... done ...')
