#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 17.09.2018 $
# by : eghbal-zadeh $
# modified by : fischer $

# --- imports -----------------------------------------------------------------
import numpy as np

result_path = r"E:\ProjectCode\VISIOMICS\trunk\BioInf\hamid\results\loc_40k"

valid_margins = [10000, 5000, 2500, 1250]

model_margins = [10000, 5000, 2500]
versions = [1, 2]
attentions = [[True, True], [False, False], [False, True]]


def prepare_str(meas, arr):
    f1mac_str = meas
    for f in arr:
        f1mac_str += '&{:.2f}'.format(f)
    f1mac_str += r"\\"
    return f1mac_str


# m_margin = 10000 #model margin
for m_margin in model_margins:
    for version in versions:
        for attention in attentions:
            first_att = attention[0]
            last_att = attention[1]

            acc, f1mic, f1mac, f1pos, precmic, precmac, precpos, recmic, recmac, recpos = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            acc[m_margin], f1mic[m_margin], f1mac[m_margin], f1pos[m_margin], precmic[m_margin], precmac[m_margin], \
            precpos[m_margin], recmic[m_margin], recmac[m_margin], recpos[
                m_margin] = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            wins_m = int(m_margin * 4 / 1000)
            for d_margin in valid_margins:
                if m_margin <= d_margin:
                    continue
                wins_d = int(d_margin * 4 / 1000)
                file = r"E:\ProjectCode\VISIOMICS\trunk\BioInf\hamid\results\loc_{}k\DeepSNP_V{}DenseNet_att{}_{}__modelmargin_{}_datamargin{}.csv" \
                    .format(wins_m, version, first_att, last_att, m_margin, d_margin)

                data = np.genfromtxt(file, dtype='str', delimiter='\n')
                # m_margin = int(file.split('\\')[-1].split('.')[0].split('__')[1].split('_')[1])
                # d_margin = int(file.split('\\')[-1].split('.')[0].split('__')[1].split('_')[2][10:])

                for i, row in enumerate(data):
                    if i < 2:
                        continue
                    if i == 2:
                        acc[m_margin][d_margin] = [float(d) for d in
                                                   row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 3:
                        f1mic[m_margin][d_margin] = [float(d) for d in
                                                     row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 4:
                        f1mac[m_margin][d_margin] = [float(d) for d in
                                                     row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 5:
                        f1pos[m_margin][d_margin] = [float(d) for d in
                                                     row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 6:
                        precmic[m_margin][d_margin] = [float(d) for d in
                                                       row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 7:
                        precmac[m_margin][d_margin] = [float(d) for d in
                                                       row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 8:
                        precpos[m_margin][d_margin] = [float(d) for d in
                                                       row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 9:
                        recmic[m_margin][d_margin] = [float(d) for d in
                                                      row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 10:
                        recmac[m_margin][d_margin] = [float(d) for d in
                                                      row.replace(';', '').replace('\\', '').split('&')[1:]]
                    elif i == 11:
                        recpos[m_margin][d_margin] = [float(d) for d in
                                                      row.replace(';', '').replace('\\', '').split('&')[1:]]

            result_array = []
            for d_margin in valid_margins:
                if m_margin <= d_margin:
                    continue
                wins_d = int(d_margin * 4 / 1000)
                result_array.append('{}k'.format(wins_d))
                result_array.append('----')

                result_array.append(prepare_str("F1 Mac", f1mac[m_margin][d_margin]))
                result_array.append(prepare_str("F1 Pos", f1pos[m_margin][d_margin]))

                result_array.append(prepare_str("Prec Mac", precmac[m_margin][d_margin]))
                result_array.append(prepare_str("Prec Pos", precpos[m_margin][d_margin]))

                result_array.append(prepare_str("Rec Mac", recmac[m_margin][d_margin]))
                result_array.append(prepare_str("Rec Pos", recpos[m_margin][d_margin]))
            out_file = r"E:\ProjectCode\VISIOMICS\trunk\BioInf\hamid\results\loc_{}k\loc_deepsnp_att{}_{}_v{}_{}k_results.txt" \
                .format(wins_m, first_att, last_att, version, wins_m)
            np.savetxt(out_file, result_array, fmt='%s')

print('Finished!')
