#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
r"""
This file contains name strings for different dimensions used in {\sc ShapNet}
"""
# =============================================================================
# Imports
# =============================================================================


from .private_methods import TensorName
from .public_methods import named_tensor_add_dim, named_tensor_get_dim, \
    named_tensor_repeat, named_tensor_split, named_tensor_squeeze, \
    named_tensor_vector_indexing_single_dim

"""
For the named tensors:
    features:  the number of individual vectors we see as a base unit
                         in Shapley value computation
    input_channels:  the dimension of the aforementioned individual vectors
    output_channels: the number of output channels
    feature_times_channels: used right before fed into the Shapley Modules

    num_passes:          the number of passes required to compute Shapley value

    num_pairs:           the number of pairs of inputs

    batch_size:          the batch size
    height:              the height of squeezed
    width:               the width of squeezed
"""

NAME_FEATURES = "features"
NAME_META_CHANNELS = "meta_channels"
# There is a difference for these two
NAME_FEATURES_META_CHANNEL = "feature_times_channels"
NAME_CHANNEL_FEATURE = "channel_times_channels"
NAME_PIXEL_CHANNEL = "pixel_times_channels"
NAME_PIXELS = "pixels"
NAME_PATCHES = "patches"
NAME_ALL = "all"

NAME_NUM_PASSES = "num_passes"

NAME_NUM_PAIRS = "num_pairs"

NAME_RESOLUTION = "resolution"
NAME_BATCH_SIZE = "batch_size"
NAME_AGG_BATCH_SIZE = "aggreated_batch_size"
NAME_HEIGHT = "height"
NAME_WIDTH = "width"

NAME_OUTPUT_CLASSES = "output_classes"
NAME_GROUPS = "groups"
