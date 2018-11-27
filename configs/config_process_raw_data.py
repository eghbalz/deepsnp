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

    # Model to train
    config.model_name = 'DeepSNP_V1'

    # Data generation parameters
    config.jitter = True

    # Training parameters
    # all default

    # Architecture specific parameters
    # all default

    # Evaluation parameters
    # only used for evaluation

    # Raw data processing parameters
    # all default

    return config
