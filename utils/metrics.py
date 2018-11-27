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
import tensorflow as tf
import keras.backend as K


def f1_score(y_true, y_pred):
    """
    Compute F1 score
    :param y_true:  true labels/groundtruth
    :param y_pred:  predicted labels
    :return:
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def precision_score(y_true, y_pred):
    """
    Compute precision score
    :param y_true:  true labels/groundtruth
    :param y_pred:  predicted labels
    :return:
    """
    p, p_op = tf.metrics.precision(labels=y_true, predictions=y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([p_op]):
        score = tf.identity(p)

    return score


def recall_score(y_true, y_pred):
    """
    Compute recall score
    :param y_true:  true labels/groundtruth
    :param y_pred:  predicted labels
    :return:
    """
    r, r_op = tf.metrics.recall(labels=y_true, predictions=y_pred)

    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([r_op]):
        score = tf.identity(r)

    return score


def get_sklearn_metrics(y_test, y_pred):
    """
    Compute SKLEARN evaluation metrics
    :param y_true:  true labels/groundtruth
    :param y_pred:  predicted labels
    :return:
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    f1_micro_sk = f1_score(y_test, y_pred, average='micro')
    f1_macro_sk = f1_score(y_test, y_pred, average='macro')
    f1_bin_sk = f1_score(y_test, y_pred, average='binary')

    precision_micro_sk = precision_score(y_test, y_pred, average='micro')
    precision_macro_sk = precision_score(y_test, y_pred, average='macro')
    precision_bin_sk = precision_score(y_test, y_pred, average='binary')

    recall_micro_sk = recall_score(y_test, y_pred, average='micro')
    recall_macro_sk = recall_score(y_test, y_pred, average='macro')
    recall_bin_sk = recall_score(y_test, y_pred, average='binary')

    acc_sk = accuracy_score(y_test, y_pred)

    return f1_micro_sk, f1_macro_sk, f1_bin_sk, precision_micro_sk, precision_macro_sk, precision_bin_sk, \
           recall_micro_sk, recall_macro_sk, recall_bin_sk, acc_sk


def write_metrics_to_txt(save_file, model_name, n_fold, loss, acc_k, acc_sk, f1_k, f1_micro_sk, f1_macro_sk, f1_bin_sk,
                         precision_k, precision_micro_sk, precision_macro_sk, precision_bin_sk, recall_k,
                         recall_micro_sk, recall_macro_sk, recall_bin_sk):
    """
    Write the computed evaluation metrics to txt file
    :param save_file:           File name and location
    :param model_name:          used model name
    :param n_fold:              number of cross validation fold
    :param loss:                optimization loss
    :param acc_k:               accuracy (keras)
    :param acc_sk:              accuracy (sklearn)
    :param f1_k:                f1 score (keras)
    :param f1_micro_sk:         f1 micro score (sklearn)
    :param f1_macro_sk:         f1 macro score (sklearn)
    :param f1_bin_sk:           f1 binary score (sklearn)
    :param precision_k:         precision (keras)
    :param precision_micro_sk:  precision micro (sklearn)
    :param precision_macro_sk:  precision macro (sklearn)
    :param precision_bin_sk:    precision binary (sklearn)
    :param recall_k:            recall (keras)
    :param recall_micro_sk:     recall micro (sklearn)
    :param recall_macro_sk:     recall macro (sklearn)
    :param recall_bin_sk:       recall binary (sklearn)
    :return:
    """
    print('WHOLE SEQ TEST FOLD {}:'.format(n_fold))
    print('Loss: {} '.format(loss))

    print('Acc K: {} '.format(acc_k))
    print('Acc SK: {} '.format(acc_sk))

    print('F1 K: {} '.format(f1_k))
    print('F1 Micro SK: {} '.format(f1_micro_sk))
    print('F1 Macro SK: {} '.format(f1_macro_sk))
    print('F1 Bin SK: {} '.format(f1_bin_sk))

    print('Precision K: {} '.format(precision_k))
    print('Precision Micro SK: {} '.format(precision_micro_sk))
    print('Precision Macro SK: {} '.format(precision_macro_sk))
    print('Precision Bin SK: {} '.format(precision_bin_sk))

    print('Recall K: {} '.format(recall_k))
    print('Recall Micro SK: {} '.format(recall_micro_sk))
    print('Recall Macro SK: {} '.format(recall_macro_sk))
    print('Recall Bin SK: {} '.format(recall_bin_sk))

    with open(save_file, "w") as text_file:
        text_file.write('MODEL {}:\n'.format(model_name))
        text_file.write('WHOLE SEQ TEST FOLD {}:\n'.format(n_fold))
        text_file.write('Loss: {} \n'.format(loss))

        text_file.write('Acc K: {} \n'.format(acc_k))
        text_file.write('Acc SK: {} \n'.format(acc_sk))

        text_file.write('F1 K: {} \n'.format(f1_k))
        text_file.write('F1 Micro SK: {} \n'.format(f1_micro_sk))
        text_file.write('F1 Macro SK: {} \n'.format(f1_macro_sk))
        text_file.write('F1 Bin SK: {} \n'.format(f1_bin_sk))

        text_file.write('Precision K: {} \n'.format(precision_k))
        text_file.write('Precision Micro SK: {} \n'.format(precision_micro_sk))
        text_file.write('Precision Macro SK: {} \n'.format(precision_macro_sk))
        text_file.write('Precision Bin SK: {} \n'.format(precision_bin_sk))
        text_file.write('Recall K: {} \n'.format(recall_k))
        text_file.write('Recall Micro SK: {} \n'.format(recall_micro_sk))
        text_file.write('Recall Macro SK: {} \n'.format(recall_macro_sk))
        text_file.write('Recall Bin SK: {} '.format(recall_bin_sk))
