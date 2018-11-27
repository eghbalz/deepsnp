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
import librosa
import numpy as np
from keras.utils import to_categorical

from network.NetRunner import NetRunner
from utils.prepare_data import get_patient_ids, plot_loc_output, load_raw_data
from utils.metrics import write_metrics_to_txt, get_sklearn_metrics


class InferenceRunner(NetRunner):
    """
    DeepSNP InferenceRunner class. Base class for inference.
    """
    def __init__(self, args=None, experiment_id=None):
        """
        Initialize InferenceRunner
        :param args:            configuration parameters
        :param experiment_id:   string identifying the experiment to run
        """
        super().__init__(args, experiment_id)

    def start_testing(self):
        """
        Start inference
        :return:
        """

        print('... starting inference ...')

        n_fold = 0
        for train_index, test_index in self.kfold.split(self.kf_X):

            test_inds = get_patient_ids(test_index, patient_indices=self.pat_ids, patient_count=self.pat_count,
                                        data_dir=self.feat_dir)

            for margin in self.margins:

                window_size = margin * 4
                model_id = '{}{}_fold{}_att{}_{}_margin{}'.format(self.model_name, self.conv_architecture, n_fold,
                                                                  self.use_input_attention, self.use_output_attention,
                                                                  margin)
                print('Model {}, Margin: {} started!'.format(model_id, margin))

                prediction_dir = os.path.join(self.save_dir, 'preds', model_id).replace('[', '').replace(']', '')
                if not os.path.exists(prediction_dir):
                    os.makedirs(prediction_dir)

                metrics_save_file = os.path.join(prediction_dir,
                                                 'model_margin{}_deepSNP_eval_metrices.txt'.format(margin))

                # only create metrics files if not already exists or localization unit output should be computed
                if not os.path.exists(metrics_save_file) or (self.plot_loc_results and self.gen_loc_output):

                    # load model
                    self.input_shape = (2, window_size, 1)
                    # get last model weights file
                    model_files = glob.glob(os.path.join(self.model_dir, model_id + '_*'))
                    if model_files:
                        model, loc_model = self.select_model(margin=margin, num_classes=self.eval_num_classes)
                        print('Model {}, Margin: {} created!'.format(model_id, margin))

                        model_losses = [float(mf.split('-')[-1][8:-3]) if 'val_loss' in mf.split('-')[-1] else np.inf
                                        for mf in model_files]
                        model_file = model_files[np.argmin(model_losses)]
                        model.load_weights(model_file)
                        print('Model {}, Margin: {} weights loaded!'.format(model_id, margin))

                    else:
                        print('Model {}, Margin {} does not exist! Skipping!'.format(model_id, margin))
                        model = None
                        loc_model = None
                        continue

                    # overall results for cross val split test data
                    y_test = []
                    pred_results = []

                    # sample loop
                    for pat_id in np.array(self.pat_id_refs)[test_inds]:
                        print('Loading sample: {}, FOLD: {}'.format(pat_id, n_fold))
                        lrr_c, baf_c, labels, delta, break_point_index = load_raw_data(self.data_dir, pat_id)

                        if 'DeepSNP' in self.model_name and self.gen_loc_output:
                            loc_test, loc_delta_test = [], []
                            point_wise_pred, point_wise_win = [], []
                        alternator = 0
                        # create windows
                        for fs in range(int(np.ceil(len(labels) / window_size))):
                            st = fs * window_size
                            ed = (fs + 1) * window_size
                            if ed > len(labels):
                                ed = len(labels)
                            eval_win = np.vstack((baf_c[st:ed], lrr_c[st:ed]))

                            real_eval_win_shape = eval_win.shape[1]

                            # pad data if smaller then 4*margin
                            if eval_win.shape[1] < window_size:
                                eval_win = np.pad(eval_win, ((0, 0), (0, window_size - eval_win.shape[1])),
                                                  'constant', constant_values=-2)

                            assert eval_win.shape[1] == window_size
                            y_raw = np.unique(labels[st:ed]).tolist()
                            if len(y_raw) > 1:
                                eval_gt = 1
                            else:
                                eval_gt = 0
                            eval_gt = to_categorical(eval_gt, self.eval_num_classes).astype(np.int32)

                            eval_bps = np.where(delta[st:ed] > 0.001)[0]

                            # append the GT for the whole patient (genome file)
                            y_test.append(eval_gt)

                            # Predict with model margin
                            x_test = eval_win[None, :, :, None]

                            y_pred = model.predict(x_test)
                            y_pred = y_pred.squeeze()
                            pred_results.append(y_pred)

                            # Get keras evaluation metrics
                            loss, acc_k, f1_k, precision_k, recall_k = model.evaluate(x_test, eval_gt[None, :],
                                                                                      batch_size=self.batch_size,
                                                                                      verbose=0)

                            if 'DeepSNP' in self.model_name and self.gen_loc_output:

                                # save prediction for each point in the window for later visualization
                                point_wise_pred.append(np.repeat(np.argmax(y_pred), real_eval_win_shape))
                                point_wise_win.append(np.ones(real_eval_win_shape) * alternator)
                                if alternator == 0:
                                    alternator = 1
                                elif alternator == 1:
                                    alternator = 0

                                # predict localization unit output with margin
                                loc_output = loc_model.predict(x_test)
                                loc_output = loc_output.squeeze()

                                # compute the delta of localisation
                                loc_delta = np.abs(librosa.feature.delta(loc_output[:, 1], width=3))

                                # Repeating the localisations
                                # rep_value = (4 * m_model) / loc_output.shape[0]
                                rep_value = real_eval_win_shape / loc_output.shape[0]
                                rep_bp_prob = np.repeat(loc_output[:, 1], rep_value)
                                rep_loc_delta = np.repeat(loc_delta, rep_value)

                                if rep_bp_prob.shape[0] < real_eval_win_shape:
                                    rep_bp_prob = np.pad(rep_bp_prob, (0, real_eval_win_shape - rep_bp_prob.shape[0]),
                                                         mode='constant', constant_values=rep_bp_prob[-1])
                                    rep_loc_delta = np.pad(rep_loc_delta,
                                                           (0, real_eval_win_shape - rep_loc_delta.shape[0]),
                                                           mode='constant', constant_values=rep_loc_delta[-1])
                                elif not rep_bp_prob.shape[0] == real_eval_win_shape:
                                    raise ValueError('Breakpoint probability shape ({}) is different to evaluation '
                                                     'window shape({})!'.format(rep_bp_prob.shape[0],
                                                                                real_eval_win_shape))

                                # Append localisation for each genomic file
                                loc_test.append(rep_bp_prob)
                                loc_delta_test.append(rep_loc_delta)

                                # plotting within the window
                                if (np.any(rep_bp_prob >= self.pred_thresh) or len(eval_bps) > 0) \
                                        and self.plot_loc_results:
                                    plot_loc_output(y_pred, rep_bp_prob, eval_gt, eval_win, eval_bps, rep_loc_delta,
                                                    prediction_dir, pat_id, margin, fs)

                        if 'DeepSNP' in self.model_name and self.gen_loc_output:
                            # concatenating the outcome of each patient (genome sample)
                            loc_prediction = np.concatenate(np.array(loc_test))
                            loc_delta_prediction = np.concatenate(np.array(loc_delta_test))

                            pwp = np.concatenate(np.array(point_wise_pred))
                            pww = np.concatenate(np.array(point_wise_win))

                            assert pwp.shape == pww.shape == loc_prediction.shape == \
                                   loc_prediction.shape == loc_delta_prediction.shape
                            out_file = os.path.join(prediction_dir,
                                                    pat_id + '_data_margin{}_deepSNP_prediction'.format(margin))

                            np.save(out_file, np.array([np.array(loc_prediction > self.pred_thresh, dtype=int),
                                                        loc_prediction, loc_delta_prediction, pwp, pww]))

                    # concatenate the predictions and GT for each FOLD
                    pred_results = np.array(pred_results)
                    y_test = np.array(y_test)

                    print(pred_results.shape, y_test.shape, pred_results[0, 0], np.min(y_test[:, 0]),
                          np.max(y_test[:, 0]))

                    y_pred = np.argmax(pred_results, 1).astype(np.int32)
                    y_test = np.argmax(y_test, 1).astype(np.int32)

                    # compute sklearn metrics
                    f1_micro_sk, f1_macro_sk, f1_bin_sk, precision_micro_sk, precision_macro_sk, precision_bin_sk, \
                    recall_micro_sk, recall_macro_sk, recall_bin_sk, acc_sk = get_sklearn_metrics(y_test, y_pred)

                    # save metrics to txt file
                    write_metrics_to_txt(metrics_save_file, model_id, n_fold, loss, acc_k, acc_sk, f1_k, f1_micro_sk,
                                         f1_macro_sk, f1_bin_sk, precision_k, precision_micro_sk, precision_macro_sk,
                                         precision_bin_sk, recall_k, recall_micro_sk, recall_macro_sk, recall_bin_sk)
                else:
                    print('Skipping {}! Already exists!'.format(metrics_save_file))

            n_fold += 1
