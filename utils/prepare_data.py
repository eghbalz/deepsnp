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
import h5py
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_patient_ids(fold_patient_indices, patient_indices, patient_count, data_dir, pat_id_filename='patient_ids.h5'):
    """
    Get patient indices of specific cross validation fold
    :param fold_patient_indices:    patient indices per fold
    :param patient_indices:         patient indices to get
    :param patient_count:           number of patients
    :param data_dir:                patient id file location
    :param pat_id_filename:         patient id file name
    :return:
    """
    h5f = h5py.File(os.path.join(data_dir, pat_id_filename), 'r')
    pat_id_refs = h5f['pat_ids'][:]
    h5f.close()

    # make sure to get all samples of all patients in the fold
    fold_pat_ids = [pid for pid, ind in patient_indices.items() if ind in fold_patient_indices]

    pat_inds = []
    for pind, pir in enumerate(pat_id_refs):
        for tpi in fold_pat_ids:
            if tpi in pir:
                pat_inds.append(pind)

    # sanity check if correct number of samples is used
    assert np.sum([patient_count[t] for t in fold_pat_ids]) == len(pat_inds)

    return pat_inds


def load_raw_data(data_dir, pat_id, file_ext='.dat'):
    """
    Load raw SNPa data from files
    :param data_dir:    directory where files are located
    :param pat_id:      patiend ID to load
    :param file_ext:    file extension (default: .dat)
    :return:
    """
    data = pd.read_csv(os.path.join(data_dir, pat_id + file_ext), sep='\t', header=0)

    # extract data from data frame
    lrr_c = np.array(data._series['LRR'])
    baf_c = np.array(data._series['BAF'])
    labels = np.array(data._series['tCN'])  # tCN
    delta = np.abs(librosa.feature.delta(labels, width=3))
    break_point_index_raw = np.where(delta > 0.001)[0]
    # fix successive breakpoints - only use one ot the tuple
    break_point_index = []
    for ind in range(len(break_point_index_raw) - 1):
        if break_point_index_raw[ind] + 1 == break_point_index_raw[ind + 1]:
            break_point_index.append(break_point_index_raw[ind])

    return lrr_c, baf_c, labels, delta, break_point_index


def plot_loc_output(y_pred, bp_prob, eval_gt, eval_win, eval_bps, loc_delta, prediction_dir, pat_id, margin,
                    win_number):
    """
    Plotting function for localization unit output
    :param y_pred:          network prediction
    :param bp_prob:         breakpoint probability
    :param eval_gt:         groundtruth label
    :param eval_win:        evaluated window (log R + BAF values)
    :param eval_bps:        evaluated breakpoints
    :param loc_delta:       delta of the localization unit output
    :param prediction_dir:  directory to save the plots too
    :param pat_id:          patient ID
    :param margin:          used margin
    :param win_number:      number of the window to plot
    :return:
    """
    window_size = margin * 4

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(eval_win[0], '.g', markersize=0.1)
    plt.ylim(ymax=-2, ymin=2)
    plt.ylabel('BAF')
    plt.title('WinSize: {} - pred/gt: {}/{} [0=no BP, 1=BP]'.format(window_size, np.argmax(y_pred), np.argmax(eval_gt)))

    plt.subplot(4, 1, 2)
    plt.plot(eval_win[1], '.g', markersize=0.1)
    plt.ylim(ymax=-3, ymin=3)
    plt.ylabel('LRR')
    plt.subplot(4, 1, 3)
    plt.plot(loc_delta, 'm', markersize=0.1)
    plt.ylim(ymax=1, ymin=0)
    for x_ in eval_bps:
        bp_ix_to_plot = x_
        plt.vlines(bp_ix_to_plot, 0, 1, colors='r', linestyles='dashed')
    plt.ylabel('LOC Delta')
    plt.subplot(4, 1, 4)
    plt.plot(bp_prob)
    plt.ylim(ymax=1, ymin=0)
    for x_ in eval_bps:
        bp_ix_to_plot = x_
        plt.vlines(bp_ix_to_plot, 0, 1, colors='r', linestyles='dashed')
    plt.ylabel('LOC Output')
    plt.tight_layout()
    # plt.show()
    png_file = os.path.join(prediction_dir, '{}_data_margin{}_win{}.png'.format(pat_id, margin, win_number))
    plt.savefig(png_file, dpi=300)

    plt.close('all')


def feature_extraction(sample_index, labels, baf, lrr, rawcopy_pred, data_shape, margin=10000, pad_val=-2):
    """
    Extract features at sample index
    :param sample_index:    sample index
    :param labels:          break point labels
    :param baf:             b-allele frequency values
    :param lrr:             log r ratio values
    :param rawcopy_pred:    rawcop predictions
    :param data_shape:      shape of the data
    :param margin:          margin to use
    :param pad_val:         padding value for windows appearing on start or end of data sequence
    :return:
    """
    window_size = margin * 4

    if sample_index < margin * 2:
        running_idx = margin * 2 - sample_index
        running_idx2 = margin * 2 + sample_index
        if running_idx2 >= len(baf):
            running_idx2 = len(baf) - 1
        ix = range(sample_index, sample_index + margin)
        baf_ix = range(0, running_idx2)
        baf_ = baf[baf_ix]
        baf = np.pad(baf_, (running_idx, 0), 'constant', constant_values=pad_val)
        lrr_ = lrr[baf_ix]
        lrr = np.pad(lrr_, (running_idx, 0), 'constant', constant_values=pad_val)
    elif sample_index + margin * 2 > data_shape[0]:
        running_idx = sample_index - margin * 2
        ix = range(sample_index - margin, data_shape[0])
        baf_ix = range(running_idx, data_shape[0])
        baf_ = baf[baf_ix]
        baf = np.pad(baf_, (0, running_idx), 'constant', constant_values=pad_val)
        lrr_ = lrr[baf_ix]
        lrr = np.pad(lrr_, (0, running_idx), 'constant', constant_values=pad_val)
    else:
        ix = range(sample_index - margin, sample_index + margin)
        baf_ix = range(sample_index - margin * 2, sample_index + margin * 2)
        baf = baf[baf_ix]
        lrr = lrr[baf_ix]

    label = []
    for l in labels[baf_ix]:
        if label == []:
            label.append(l)
        elif l != label[-1]:
            label.append(l)

    rc_pred = []
    for l in rawcopy_pred[baf_ix]:
        if rc_pred == []:
            rc_pred.append(l)
        elif l != label[-1]:
            rc_pred.append(l)

    assert baf.shape[0] == window_size
    assert lrr.shape[0] == window_size

    feat = np.vstack((baf, lrr))

    return feat, rc_pred, label, ix


def feature_extraction_jitter(sample_index, labels, baf, lrr, rawcopy_pred, data_shape, margin=10000):
    """
    Extract features at sample index and jitter windows randomly
    :param sample_index:    sample index
    :param labels:          break point labels
    :param baf:             b-allele frequency values
    :param lrr:             log r ratio values
    :param rawcopy_pred:    rawcop predictions
    :param data_shape:      shape of the data
    :param margin:          margin to use
    :return:
    """
    window_size = margin * 4
    min_margin = int(margin / 2)
    if data_shape[0] - window_size >= sample_index >= min_margin:
        starting_pt = max(0, sample_index - window_size + min_margin)
        ending_pt = sample_index - min_margin

        try:
            running_idx = np.random.choice(range(starting_pt, ending_pt))
        except ValueError:
            print(starting_pt)

        running_idx2 = running_idx + window_size

        baf_ix = range(running_idx, running_idx2)
        baf = baf[baf_ix]
        lrr = lrr[baf_ix]

        label = []
        for l in labels[baf_ix]:
            if label == []:
                label.append(l)
            elif l != label[-1]:
                label.append(l)

        rc_pred = []
        for l in rawcopy_pred[baf_ix]:
            if rc_pred == []:
                rc_pred.append(l)
            elif l != label[-1]:
                rc_pred.append(l)

        assert baf.shape[0] == window_size
        assert lrr.shape[0] == window_size

        feat = np.vstack((baf, lrr))
    else:
        feat = None
        rc_pred = None
        label = None

    return feat, rc_pred, label


def plot_lrr_baf(data, ix, data_type_names, feat=None, title=None, save_file=None):
    """
    Plotting function for log R ratio and b-allele frequency
    :param data:            pandas data frame containing data
    :param ix:              data index to plot
    :param data_type_names: data keys - headers of pandas file
    :param feat:            features
    :param title:           plotting title
    :param save_file:       save file path and filename
    :return:
    """
    plt.close('all')
    for data_type_ix in [2, 3, 4]:
        dmy = np.array(data._series[data_type_ix])[ix]
        plt.subplot(2, 2, data_type_ix - 1)
        plt.grid(False)
        plt.title(data_type_names[data_type_ix])
        plt.plot(dmy, '.')
    if feat is not None:
        plt.subplot(2, 2, 4)
        plt.grid(False)
        plt.imshow(feat, aspect='auto', cmap='viridis')
    if title is not None:
        plt.title(title)
    if save_file is not None:
        if not os.path.exists(os.path.dirname(save_file)):
            os.mkdir(os.path.dirname(save_file))
        if os.path.exists(save_file):
            raise FileExistsError
        plt.savefig(save_file, dpi=300)

    # plt.show(False)
    # plt.pause(1)
    # plt.draw()
    plt.clf()
