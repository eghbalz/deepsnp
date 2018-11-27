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
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import to_categorical


def prep_lists(inputs, targets, classes, shuffle):
    """
    Prepare lists
    :param inputs:      inputs
    :param targets:     targets
    :param classes:     classes
    :param shuffle:     shuffle lists
    :return:
    """
    targets = np.argmax(targets, axis=1).astype('int32')
    nclass = len(classes)
    ninputs = []
    nindices = []
    ntargets = []
    nlen = []
    nix = []
    for cls in classes:
        ix = np.where(targets == cls)[0]
        nix.append(ix)
        if shuffle:
            np.random.shuffle(ix)
        ninputs.append(inputs[ix])
        nindices.append(np.arange(len(inputs)))
        ntargets.append(targets[ix])
        nlen.append(len(ix))

    return targets, nclass, ninputs, nindices, ntargets, nlen, nix


class StratifiedImageGenerator(ImageDataGenerator):
    """
    Stratified image generator. Inheriting from Keras ImageDataGenerator class
    """
    @staticmethod
    def stratified_iterator(inputs, targets=None, batch_size=32, shuffle=True, forever=True, classes=[]):
        """
        Stratified image iterator
        :param inputs:      input array
        :param targets:     target array
        :param batch_size:  batch size
        :param shuffle:     shuffle data
        :param forever:     run continuously
        :param classes:     classes
        :return:
        """
        targets = np.argmax(targets, axis=1).astype('int32')
        targets, nclass, ninputs, nindices, ntargets, nlen, nix = prep_lists(inputs, targets, classes, shuffle)

        while True:

            excerpts = []
            nexcerpts = []
            for i, cls in enumerate(classes):
                excerpt = []
                for start_idx in range(0, nlen[i] - int(batch_size / nclass) + 1, int(batch_size / nclass)):
                    ix_ = slice(start_idx, start_idx + int(batch_size / nclass))
                    ex = nix[i][ix_]
                    excerpt.append(ex)
                excerpts.append(excerpt)
                nexcerpts.append(len(excerpt))

            min_nexpt = min(nexcerpts)
            max_nexpt = max(nexcerpts)

            for ex in range(min_nexpt):
                final_excerpt = []
                for i, cls in enumerate(classes):
                    final_excerpt.append(excerpts[i][ex])
                final_excerpt_ = np.concatenate(final_excerpt)
                yield inputs[final_excerpt_], to_categorical(targets[final_excerpt_], nclass)
            if not forever:
                break


class BinaryStratifiedImageGenerator(ImageDataGenerator):
    """
    Binary stratified image generator. Inheriting from Keras ImageDataGenerator class
    """
    @staticmethod
    def stratified_iterator(inputs, targets=None, batch_size=32, shuffle=True, forever=True):
        """
        Stratified image iterator
        :param inputs:      input array
        :param targets:     target array
        :param batch_size:  batch size
        :param shuffle:     shffle data
        :param forever:     run continuously
        :return:
        """

        classes = np.unique(targets)
        targets, nclass, ninputs, nindices, ntargets, nlen, nix = prep_lists(inputs, targets, classes, shuffle)

        while True:

            excerpts = []
            nexcerpts = []
            for i, cls in enumerate(classes):
                excerpt = []
                for start_idx in range(0, nlen[i] - int(batch_size / nclass) + 1, int(batch_size / nclass)):
                    ix_ = slice(start_idx, start_idx + int(batch_size / nclass))
                    ex = nix[i][ix_]
                    excerpt.append(ex)
                excerpts.append(excerpt)
                nexcerpts.append(len(excerpt))

            min_nexpt = min(nexcerpts)
            max_nexpt = max(nexcerpts)

            for ex in range(min_nexpt):
                final_excerpt = []
                for i, cls in enumerate(classes):
                    final_excerpt.append(excerpts[i][ex])
                final_excerpt_ = np.concatenate(final_excerpt)
                return inputs[final_excerpt_], targets[final_excerpt_]
            if not forever:
                break
