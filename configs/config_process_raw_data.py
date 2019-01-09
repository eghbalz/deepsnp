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
    config = ConfigFlags()

    # Directories
    config.data_dir =   r'/path/to/raw_data'

    # Model to train
    config.model_name = 'process_raw_data'

    # Data generation parameters
    config.jitter =     True

    # Training parameters
    # all default

    # Architecture specific parameters
    # all default

    # Evaluation parameters
    # only used for evaluation

    # Raw data processing parameters
    # all default

    return config
