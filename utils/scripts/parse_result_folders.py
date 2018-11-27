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

    # DATA_PATH = r'E:\Projects\VISIOMICS\trunk\BioInf\results\preds'
    DATA_PATH = r'S:\Project_Stuff\VISIOMICS\DeepSNP\std_eval_metrics'
    MET_RES_PATH = os.path.join(DATA_PATH, 'metric_results')
    if not os.path.exists(MET_RES_PATH):
        os.mkdir(MET_RES_PATH)

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
            TXT_FILE = TXT_FILE[0]
            # print('Parsing File: {}'.format(TXT_FILE))
            txt_dir = os.path.dirname(TXT_FILE)
            txt_filename = os.path.basename(TXT_FILE)
            win_size = int(txt_filename.split('_')[1][6:])
            fold_ind = txt_dir.find('fold')
            fold = int(txt_dir[fold_ind + 4:fold_ind + 5])
            model_name = os.path.basename(txt_dir[:fold_ind - 1])
            if 'DeepSNP' in model_name:
                # also get which version it is (no, final, full attention)
                att_ind = txt_dir.find('att')
                marg_ind = txt_dir.find('margin')
                model_name = model_name + '_' + txt_dir[att_ind:marg_ind]

            if model_name not in metric.keys():
                metric[model_name] = {}
            if win_size not in metric[model_name].keys():
                metric[model_name][win_size] = {}
            if fold not in metric[model_name][win_size].keys():
                metric[model_name][win_size][fold] = {}
            with open(os.path.join(DATA_PATH, TXT_FILE), 'r') as in_file:
                for line in in_file:
                    # run through file and get all info
                    for met in metrics:
                        if met in line:
                            l = re.search(met, line)
                            metric[model_name][win_size][fold][met] = float(line[l.end():])
                            break

    n_models_total = len(metric[list(metric.keys())[0]])

    # run through models in metric dict and create result csv

    for model_name, model_items in metric.items():
        CSV_FILE = os.path.join(MET_RES_PATH, model_name + '.csv')

        with open(os.path.join(DATA_PATH, CSV_FILE), 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=';')

            for margin, margin_items in model_items.items():
                writer.writerow(['WinSize: {} - Margin: {}'.format(int(margin * 4), margin), *header_folds])
                # run through folds
                met_strings = dict()
                for fold, fold_items in margin_items.items():
                    for m_k, m_i in fold_items.items():
                        if m_k not in met_strings:
                            met_strings[m_k] = [m_i]
                        else:
                            met_strings[m_k].append(m_i)

                for met, p_met in zip(metrics, print_metrics):
                    ms = [[m * 100, '&'] if i < n_folds_total - 1 else [m * 100, r'\\'] for i, m in
                          enumerate(met_strings[met])]
                    ms = list(chain.from_iterable(ms))
                    writer.writerow([p_met, '&', *ms])

                writer.writerow('')
