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
from configs.config import ConfigFlags


def load_config():
    config = ConfigFlags().return_flags()

    # Directories
    config.data_dir = r'S:\Project_Stuff\VISIOMICS\Data\Genomicdata_26072018_samples50'
    # config.save_dir = r'E:\Projects\VISIOMICS\trunk\BioInf\DeepSNP\results'
    # config.model_dir = r'E:\Projects\VISIOMICS\trunk\BioInf\DeepSNP\models'
    # config.model_dir = r'S:\Project_Stuff\VISIOMICS\DeepSNP\models'
    # config.log_dir = r'E:\Projects\VISIOMICS\trunk\BioInf\DeepSNP\logs'

    # Model to train
    config.model_name = 'DeepSNP_V2'

    # Data generation parameters
    config.jitter = True

    # Training parameters
    # all default

    # Architecture specific parameters
    # all default

    # Evaluation parameters
    # only used for evaluation

    return config
