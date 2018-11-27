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

from network.NetRunner import NetRunner
from utils.metrics import get_sklearn_metrics, write_metrics_to_txt


class RawCopyRunner(NetRunner):
    """
    DeepSNP RawCopyRunner class. Base class for Rawcopy prediction evaluation.
    """
    def __init__(self, args=None, experiment_id=None):
        """
        Initialize RawCopyRunner
        :param args:            configuration parameters
        :param experiment_id:   string identifying the experiment to run
        """
        super().__init__(args, experiment_id)

    def evaluate(self):
        """
        Evaluate Rawcopy predictions
        :return:
        """

        print('... starting rawcopy evaluation ...')

        n_fold = 0
        for train_index, test_index in self.kfold.split(self.kf_X):
            for margin in self.margins:

                y_test, y_pred = self.load_test_rc_data(test_index, margin)

                y_pred = np.argmax(y_pred, 1).astype(np.int32)
                y_test = np.argmax(y_test, 1).astype(np.int32)

                # compute sklearn metrics
                f1_micro_sk, f1_macro_sk, f1_bin_sk, precision_micro_sk, precision_macro_sk, precision_bin_sk, \
                recall_micro_sk, recall_macro_sk, recall_bin_sk, acc_sk = get_sklearn_metrics(y_test, y_pred)

                # save metrics to txt file
                print('WHOLE SEQ TEST FOLD {}:'.format(n_fold))
                print('WINSIZE: {}, MARGIN: {}'.format(margin * 4, margin))
                print('Acc SK: {} '.format(acc_sk))

                print('F1 Micro SK: {} '.format(f1_micro_sk))
                print('F1 Macro SK: {} '.format(f1_macro_sk))
                print('F1 Bin SK: {} '.format(f1_bin_sk))

                print('Precision Micro SK: {} '.format(precision_micro_sk))
                print('Precision Macro SK: {} '.format(precision_macro_sk))
                print('Precision Bin SK: {} '.format(precision_bin_sk))

                print('Recall Micro SK: {} '.format(recall_micro_sk))
                print('Recall Macro SK: {} '.format(recall_macro_sk))
                print('Recall Bin SK: {} '.format(recall_bin_sk))

            n_fold += 1
