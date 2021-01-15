#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
This file is used to initialize the utilities for constructing ShapNets
"""
# =============================================================================
# Imports
# =============================================================================


from .dims import ListModuleDimensions, ModuleDimensions, process_list_sizes
from .named_tensor_utils import NAME_AGG_BATCH_SIZE, NAME_ALL, NAME_BATCH_SIZE, \
    NAME_CHANNEL_FEATURE, NAME_FEATURES, NAME_FEATURES_META_CHANNEL, \
    NAME_GROUPS, NAME_HEIGHT, NAME_META_CHANNELS, NAME_META_CHANNELS, \
    NAME_NUM_PAIRS, NAME_NUM_PASSES, NAME_OUTPUT_CLASSES, NAME_PATCHES, \
    NAME_PIXELS, NAME_PIXEL_CHANNEL, NAME_RESOLUTION, NAME_WIDTH, \
    named_tensor_add_dim, named_tensor_get_dim, named_tensor_repeat, \
    named_tensor_split, named_tensor_vector_indexing_single_dim
from .utils import flatten, generate_binary_sequence, get_indices
