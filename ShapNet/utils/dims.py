#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
In this module we create a class that deal with the different dimensions
"""

# =============================================================================
# Imports
# =============================================================================
from copy import deepcopy
from typing import List, Tuple


# =============================================================================
# Utility Functions
# =============================================================================
def _element_process(element: int or Tuple[int, int]) -> Tuple[int, int]:
    """
    get the tuple version of it
    Args:
        element (): the element of the sizes

    Returns: tuple of ints indicating sizes

    """
    if isinstance(element, int):
        return element, element
    else:
        return element


def process_list_sizes(
        list_sizes: List[int or Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
    """
    Here we process the list s.t. it is easier to deal with in the presence
    of juxtaposition.
    This is useful later for assigning the attributes of nn.Unfold & nn.Fold

    Args:
        list_sizes (): reform the list of elements to List[Tuple[int, int]]

    Returns:

    """
    for i, size in enumerate(list_sizes):
        list_sizes[i] = _element_process(element=size)

    return list_sizes


# =============================================================================
# Classes
# =============================================================================
class ModuleDimensions(object):
    """
    This class contains the dimensions inside a ShapleyModule

    Attributes:
        features (int): the number of input features to compute
        importance values for
        in_channel (int): the dimension of a single input feature
        out_channel (int): the number of output desired for this model

    """

    def __init__(self, features: int, in_channel: int, out_channel: int):
        """
        Initialize the the instance
        Args:
            features (): as seen in class Docstring
            in_channel (): as seen in class Docstring
            out_channel (): as seen in class Docstring
        """
        super(ModuleDimensions, self).__init__()
        self.features = features
        self.in_channel = in_channel
        self.out_channel = out_channel

        assert isinstance(self.in_channel, int)
        assert isinstance(self.in_channel, int)
        assert isinstance(self.out_channel, int)


class ListModuleDimensions(ModuleDimensions):
    """
    A list of `ModuleDimensions`
    This is useful for Deep ShapNet's instantiation, specifically for
    tracking the dimensions of the Shapley representation.
    """

    def __init__(
            self,
            features=None,
            in_channel: int = None,
            out_channel: int = None,
            list_output_channels: List[int or Tuple[int, int]] = None
    ):
        """

        Args:
            features ():                the number of input features
            in_channel ():              the number of input channels
            out_channel ():             the number of output channels
            list_output_channels ():    the list of outputs
        """
        super(ListModuleDimensions, self).__init__(
            features=features,
            in_channel=in_channel,
            out_channel=out_channel,
        )
        # create the proper list for the dimensions 
        list_output_channels = deepcopy(list_output_channels)
        list_output_channels.insert(0, in_channel)

        # append the modules individually to the module list
        self.dimension_list = []
        for i in range(len(list_output_channels) - 1):
            num_channels = list_output_channels[i]
            if isinstance(num_channels, tuple):
                self.dimension_list.append(
                    ModuleDimensions(
                        features, in_channel=num_channels[0],
                        out_channel=num_channels[1]
                    )
                )
            else:
                self.dimension_list.append(
                    ModuleDimensions(
                        features, in_channel=num_channels,
                        out_channel=list_output_channels[i + 1]
                    )
                )

    def __getitem__(self, item):
        return self.dimension_list[item]
