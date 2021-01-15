#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
Utility functions
"""
# =============================================================================
# Imports
# =============================================================================
from typing import Any, List, Tuple

import torch


def _place_holder(x: Any):
    """
    Whatever in, whatever out

    Args:
        x ():

    Returns:

    """
    return x


def get_indices(
        total_num: int,
        index_of_interest: int,
        index_in_index: int
) -> Tuple[int, int]:
    r"""
    To get the id of the first variable of different test_modules,
    the second variable is easy to get once we have that of the first
    variable
    Args:
        total_num (): the index of the stages as of now
        index_of_interest (): the index of the test_module
        index_in_index (): the index inside the index of interest

    Returns:
        the index of the first variable once given the stage id
         and the test_module id of the current test_module

    """
    remainder = total_num - index_of_interest - 1
    group = 2 ** total_num / 2 ** remainder

    base = index_in_index * 2 // group * group
    add = index_in_index * 2 % group / 2

    index_1 = int(base + add)
    index_2 = index_1 + 2 ** index_of_interest  # get the second index

    return index_1, index_2


def flatten(x: List[List[Any]], function: callable = _place_holder) \
        -> List[Any]:
    """

    Args:
        x ():
        function ():

    Returns:

    """
    return [function(item) for sublist in x for item in sublist]


def generate_binary_sequence(size) -> torch.Tensor:
    r"""
    Generate binary sequences of size of choice
    :return: all the possible binary sequences
            of shape (2 ** size, size)
    """

    # Same as before, we define an inner recursive function
    def _gen(sequence_possessed: torch.Tensor, _size: int) -> torch.Tensor:
        r"""
        Recursively generate binary sequences
        :param sequence_possessed: the already have sequence
        :param _size: the target size of the sequence
        :return:
        """
        if _size == sequence_possessed.shape[1]:
            return sequence_possessed
        base = sequence_possessed.repeat([2, 1])
        appendix = torch.cat(
            [torch.ones(base.shape[0] // 2, 1), torch.zeros(
                base.shape[0] // 2, 1)], dim=0)
        return _gen(torch.cat([base, appendix], dim=1), _size)

    # now we call the function
    binary_sequence_generated = _gen(
        torch.Tensor([[1.], [0.]]), size).float().unsqueeze(-1)

    return binary_sequence_generated
