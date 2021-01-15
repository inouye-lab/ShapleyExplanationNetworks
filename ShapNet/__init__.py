#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
r"""
This file is used to initialize the {\sc ShapNet}
"""
# =============================================================================
# Imports
# =============================================================================


from .basic import ShapleyModule, ShapleyNetwork
from .deep import DeepButterflyShapleyNetwork, DeepShapleyNetwork
from .shallow import GeneralizedAdditiveModel, \
    OverlappingShallowShapleyNetwork, ShallowShapleyNetwork
from .vision import DeepConvShapNet, ShallowConvShapleyNetwork
