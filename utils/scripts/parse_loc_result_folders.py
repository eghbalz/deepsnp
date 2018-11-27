#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 20.08.2018 $
# by : fischer $

# --- imports -----------------------------------------------------------------
import csv
import os
import re
from glob import glob
from itertools import chain

if __name__ == '__main__':

    DATA_PATH = r'E:\Projects\VISIOMICS\trunk\BioInf\results\loc_eval\preds'
    RES_PATH = r'E:\Projects\VISIOMICS\trunk\BioInf\hamid\results'
    # MET_RES_PATH = os.path.join(DATA_PATH, 'metric_results')
    # if not os.path.exists(MET_RES_PATH):
    #     os.mkdir(MET_RES_PATH)

    metrics = ['Loss: ', 'Acc SK: ', 'F1 Micro SK: ', 'F1 Macro SK: ', 'F1 Bin SK: ', 'Precision Micro SK: ',
               'Precision Macro SK: ', 'Precision Bin SK: ', 'Recall Micro SK: ', 'Recall Macro SK: ',
               'Recall Bin SK: ']
    print_metrics = ['Loss', 'Acc', 'F1 Mic', 'F1 Mac', 'F1 Pos', 'Prec Mic', 'Prec Mac', 'Prec Pos', 'Rec Micro',
                     'Rec Mac', 'Rec Pos']

    res_folders = glob(os.path.join(DATA_PATH, '*'))

    n_folds_total = 6
    header_folds = [['Fold {}'.format(i), '&'] if i < n_folds_total - 1 else ['Fold {}'.format(i), r'\\'] for i in
                    range(n_folds_total)]
    header_folds = ['&'] + list(chain.from_iterable(header_folds))

    # run through result folders and load metric results
    metric = {}
    for rf in res_folders:
        TXT_FILE = glob(os.path.join(rf, '*.txt'))

        if TXT_FILE:
            for txt in TXT_FILE:
                # print('Parsing File: {}'.format(TXT_FILE))
                txt_dir = os.path.dirname(txt)
                txt_filename = os.path.basename(txt)
                win_size = int(txt_filename.split('_')[1][6:])
                data_margin = int(os.path.basename(txt).split('_')[1][6:])
                assert win_size == data_margin
                model_margin = int(os.path.basename(txt).split('_')[3][6:])
                fold_ind = txt_dir.find('fold')
                fold = int(txt_dir[fold_ind + 4:fold_ind + 5])
                model_name = os.path.basename(txt_dir[:fold_ind - 1])
                if 'DeepSNP' in model_name:
                    # also get which version it is (no, final, full attention)
                    att_ind = txt_dir.find('att')
                    marg_ind = txt_dir.find('margin')
                    model_name = model_name + '_' + txt_dir[att_ind:marg_ind]

                # initialize dict
                if model_name not in metric.keys():
                    metric[model_name] = {}
                if model_margin not in metric[model_name].keys():
                    metric[model_name][model_margin] = {}
                if data_margin not in metric[model_name][model_margin].keys():
                    metric[model_name][model_margin][data_margin] = {}
                if fold not in metric[model_name][model_margin][data_margin].keys():
                    metric[model_name][model_margin][data_margin][fold] = {}

                with open(os.path.join(DATA_PATH, txt), 'r') as in_file:
                    for line in in_file:
                        # run through file and get all info
                        for met in metrics:
                            if met in line:
                                l = re.search(met, line)
                                try:
                                    met_value = float(line[l.end():])
                                except:
                                    met_value = None
                                metric[model_name][model_margin][data_margin][fold][met] = met_value
                                break

    n_models_total = len(metric[list(metric.keys())[0]])

    # run through models in metric dict and create result csv

    for model_name, model_items in metric.items():
        for m_margin, m_margin_items in model_items.items():
            for d_margin, d_margin_items in m_margin_items.items():
                CSV_FILE = '{}_modelmargin_{}_datamargin{}.csv'.format(model_name, m_margin, d_margin)

                with open(os.path.join(RES_PATH, 'loc_{}k'.format(int(m_margin * 4 / 1000)), CSV_FILE), 'w',
                          newline='') as out_file:
                    writer = csv.writer(out_file, delimiter=';')

                    writer.writerow(
                        ['MODEL: WinSize: {} - Margin: {}'.format(int(m_margin * 4), m_margin), *header_folds])
                    writer.writerow(
                        ['DATA: WinSize: {} - Margin: {}'.format(int(d_margin * 4), d_margin), *header_folds])
                    # run through folds
                    met_strings = dict()
                    for fold, fold_items in d_margin_items.items():
                        for m_k, m_i in fold_items.items():
                            if m_k not in met_strings:
                                met_strings[m_k] = [m_i]
                            else:
                                met_strings[m_k].append(m_i)

                    for met, p_met in zip(metrics, print_metrics):
                        if None not in met_strings[met]:
                            ms = [[m * 100, '&'] if i < n_folds_total - 1 else [m * 100, r'\\'] for i, m in
                                  enumerate(met_strings[met])]
                            ms = list(chain.from_iterable(ms))
                            writer.writerow([p_met, '&', *ms])

                    writer.writerow('')
