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
from itertools import chain

if __name__ == '__main__':

    DATA_PATH = r'S:\UserHome\fischer\VISIOMICS'
    TXT_FILE = r'slurm.goldberg.220.out'
    CSV_FILE = r'DeepSNPv1_fullAtt_MulitWin.goldberg.220.results.csv'
    # CSV_FILE = os.path.splitext(TXT_FILE)[0] + '.csv'

    metrics = ['Loss: ', 'Acc SK: ', 'F1 Micro SK: ', 'F1 Macro SK: ', 'F1 Bin SK: ', 'Precision Micro SK: ',
               'Precision Macro SK: ', 'Precision Bin SK: ', 'Recall Micro SK: ', 'Recall Macro SK: ',
               'Recall Bin SK: ', 'Train y: ']

    fold_cnt = 0
    metric = dict()
    with open(os.path.join(DATA_PATH, TXT_FILE), 'r') as in_file:
        for line in in_file:
            # run through file and get all info

            # model description to get window size
            if 'image_input (InputLayer) ' in line:
                m = re.search('None, 2, \\d+,', line)
                win_size = int(re.findall(r'\b\d+\b', line[m.start():m.end()])[1])

                if win_size not in metric:
                    metric[win_size] = []

            # beginning of result block
            elif 'WHOLE SEQ TEST FOLD' in line:
                n_fold = int(re.findall(r'\b\d+\b', line)[0])
                fold_cnt += 1

            elif fold_cnt > 0:
                if len(metric[win_size]) < fold_cnt:
                    metric[win_size].append(dict())
                for met in metrics:
                    if met in line:
                        l = re.search(met, line)
                        if met == 'Train y: ':
                            metric[win_size][n_fold]['data_split'] = line[:-1]
                        else:
                            metric[win_size][n_fold][met] = float(line[l.end():])
                        break

    n_folds_total = len(metric[list(metric.keys())[0]])
    header_folds = [['Fold {}'.format(i), '&'] if i < n_folds_total - 1 else ['Fold {}'.format(i), r'\\'] for i in
                    range(n_folds_total)]
    header_folds = list(chain.from_iterable(header_folds))
with open(os.path.join(DATA_PATH, CSV_FILE), 'w', newline='') as out_file:
    writer = csv.writer(out_file, delimiter=';')

    for key, items in metric.items():
        writer.writerow(['WinSize: {} - Margin: {}'.format(key, int(key / 4)), *header_folds])
        # run through folds
        met_strings = dict()
        for fold in items:
            for m_k, m_i in fold.items():
                if m_k not in met_strings:
                    met_strings[m_k] = [m_i]
                else:
                    met_strings[m_k].append(m_i)

        for met in metrics:
            if met is 'Train y: ':
                met = 'data_split'

            ms = [[m, '&'] if i < n_folds_total - 1 else [m, r'\\'] for i, m in enumerate(met_strings[met])]
            ms = list(chain.from_iterable(ms))
            writer.writerow([met, *ms])

        writer.writerow('')
