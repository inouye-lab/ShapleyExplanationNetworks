#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
Basic methods to composite later
"""
# =============================================================================
# Imports
# =============================================================================

from typing import List, Tuple

import torch

# =============================================================================
# constants
# =============================================================================
TensorName = str  # Just used to not confuse anyone


# =============================================================================
# functions
# =============================================================================
def dummy(any, *args, **kwargs):
    """
    A dummy place holder callable
    Args:
        any (): anything

    Returns:
        input
    """
    return any


def name_insert(names: Tuple[TensorName], index: int, name: TensorName,
                *args, **kwargs) -> List[TensorName]:
    """
    insert a name

    Args:
        names (): the names to insert into
        index (): the index at which to insert the name
        name (): the name to insert

    Returns:
        inserted name list

    """
    names = list(names)
    if index == -1:
        index = len(names)
    elif index < 0:
        index = index + 1
    names.insert(index, name)
    return names


def add_dim(tensor: torch.Tensor, index: int,
            *args, **kwargs) -> torch.Tensor:
    """
    A wrapper function for torch.squeeze

    Args:
        tensor (): the tensor to unsqueeze
        index (): the index of to unsqueeze

    Returns:
        The unsqueeze tensor
    """
    return tensor.unsqueeze(index)
