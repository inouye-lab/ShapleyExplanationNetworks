#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
This module contains the construction of Deep ShapNets as described
in the paper with other variants used in the experiments
"""
# =============================================================================
# Imports
# =============================================================================
from abc import ABC
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import ModuleList

from .basic import ShapleyModule, ShapleyNetwork
from .shallow import ShallowShapleyNetwork
from .utils import ModuleDimensions, NAME_FEATURES, \
    NAME_META_CHANNELS, get_indices, named_tensor_get_dim
from .utils.named_tensor_utils import TensorName


# =============================================================================
# Classes
# =============================================================================
class DeepShapleyNetwork(ShapleyNetwork, ABC):
    r"""
    Deep {\sc ShapNet} as described in paper

    Notes:
        the concept of stage is formulated as a procedure in which intermediate
        shapley values are computed for once

    Attributes:
        num_stages (int): the number of stages computed from features
        input features at different stages

    """

    def __init__(
            self,
            list_shapnets: List[ShallowShapleyNetwork] = None,
            reference_values: torch.Tensor = None,
            named_output: bool = False,
            residual: bool = True,
            pruning: List[float] or float = 0.,
    ):
        r"""
        Initialize the Deep {\sc ShapNet}
        Args:
            list_shapnets ():
                Only support list of lambda ShallowShapleyNetwork, but could be
                overloaded in future inherited classes

            reference_values (): the reference values for each of the input
                features
            named_output (): if the output will be named
            residual (): if use residual connection
            pruning (): the fraction of which to be pruned out
        """
        dimensions = ModuleDimensions(
            features=list_shapnets[0].dimensions.features,
            in_channel=list_shapnets[0].dimensions.in_channel,
            out_channel=list_shapnets[-1].dimensions.out_channel
        )
        super(DeepShapleyNetwork, self).__init__(
            dimensions=dimensions,
            reference_values=reference_values,
            named_output=named_output,
        )

        # compute the number of stages
        if isinstance(list_shapnets, ModuleList):
            self.list_shapnets = list_shapnets
        else:
            self.list_shapnets = ModuleList(list_shapnets)
        self.num_stages = len(self.list_shapnets)

        # keep track of the dimensions of each stages
        self.input_meta_channels = [
            module_dict.dimensions.in_channel
            for module_dict in self.list_shapnets
        ]

        self.residual = residual
        if isinstance(pruning, float):
            self.pruning = [pruning for _ in range(len(list_shapnets))]
        elif isinstance(pruning, list):
            self.pruning = pruning

        self.list_shapnets[0].align_reference_values(reference_values)

    def _in_stage_forward(
            self,
            id_stage: int,
            stage_layer: ShallowShapleyNetwork,
            shapley_previous: torch.Tensor,
            batch_size: int,
            dim_input_channels: int
    ) -> torch.Tensor:
        r"""
        This is the in-stage forward pass,
            which could be overloaded with shared parameters
        Args:
            id_stage (): the index of the current stage
            stage_layer(): the Shallow {\sc ShapNet}
            shapley_previous (): inputs to the Shallow {\sc ShapNet}
            batch_size (): the number of samples in the batch
            dim_input_channels (): the number of input channels

        Returns:
            the Shapley representation

        """
        # placeholder

        shapley_previous, _ = stage_layer.explained_forward(shapley_previous)

        return shapley_previous

    def explained_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        As in Shapley Base

        Notes:
            This method loops for each different stage

        Args:
            inputs (): prepared inputs

        Returns:
            of shape (batch_size, features, output_channels)
        """
        batch_size = inputs.shape[0]
        shapley_values = inputs
        # For each stage
        for id_stage, (stage_layer, in_channel) in enumerate(
                zip(self.list_shapnets, self.input_meta_channels)
        ):
            shapley_values = self.in_stage_process(
                shapley_values, id_stage, stage_layer, batch_size, in_channel
            )

        shapley_values = self._final_process(
            shapley_values, len(self.list_shapnets))

        return shapley_values

    def _final_process(
            self, shapley_values: torch.Tensor, id_stage: int, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Depending on the usage, this method is for final processing before
        output the values

        Args:
            shapley_values (): the shapley values computed from the last stage
            args (): placeholder
            kwargs (): placeholder

        Returns:
            Shapley values final for presenting

        """
        return shapley_values

    def progression(
            self,
            x: torch.Tensor,
            mem_efficient: bool = False,
    ) -> List[torch.Tensor]:
        r"""
        Differs with base method in that it records every layer's output
        Args:
            x (): the input
            mem_efficient (): if to do a mem_efficient method

        Returns:
            a list of tensors that are the intermediate representations.

        """
        x = self._prepare_input_dim(
            x, self.dimensions.in_channel, explain=True)
        batch_size = x.shape[0]
        shapley_values = x
        progression = list()
        for id_stage, (stage_layer, input_meta_channel) in enumerate(
                zip(self.list_shapnets, self.input_meta_channels)
        ):
            shapley_values = self.in_stage_process(
                shapley_values, id_stage, stage_layer, batch_size,
                input_meta_channel
            )
            if mem_efficient:
                progression.append(shapley_values.clone().cpu())
            else:
                progression.append(shapley_values.clone())

        return progression

    def in_stage_process(
            self,
            shapley_values: torch.Tensor,
            id_stage: int,
            stage_layer: ShallowShapleyNetwork,
            batch_size: int,
            input_meta_channel: int
    ) -> torch.Tensor:
        r"""
        The overall process in one stage including forward, residual and
        pruning
        Args:
            shapley_values (): the shapley values of the previous output
            id_stage (): the index of the current stage
            stage_layer (): the current Shallow layer
            batch_size (): the back size of the current representation
            input_meta_channel (): the channel of the input representations

        Returns:
            the shapley representation

        """
        output_shapley = self._in_stage_forward(
            id_stage=id_stage, stage_layer=stage_layer,
            shapley_previous=shapley_values, batch_size=batch_size,
            dim_input_channels=input_meta_channel
        )  # Now within each stage
        shapley_values = self.residual_process(output_shapley, shapley_values)
        shapley_values = self.prune(shapley_values, id_stage)
        return shapley_values

    def residual_process(
            self,
            output_shapley: torch.Tensor,
            shapley_values: torch.Tensor
    ) -> torch.Tensor:
        r"""
        The residual connection if desired and possible

        Args:
            output_shapley (): the Shapley representation computed for the 
                current stage
            shapley_values (): the Shapley representation of the previous 
                stage

        Returns:
            Shapley representation

        """
        if self.residual and \
                shapley_values.shape[1] == output_shapley.shape[1]:
            return output_shapley + shapley_values
        else:
            return output_shapley

    def prune(
            self, shapley_values: torch.Tensor, id_stage: int
    ) -> torch.Tensor:
        r"""
        Prune the output of a stage

        Args:
            id_stage (): the index of the stage
            shapley_values (): the current stage's pre-pruning output

        Returns:
            the pruned Shapley representation

        """
        # First check if we need pruning in the first place
        pruning = self.pruning[id_stage]
        num_features = named_tensor_get_dim(shapley_values, NAME_FEATURES)
        k = int(num_features * pruning)  # the number to prune
        if pruning * k == 0:
            return shapley_values

        names = shapley_values.names
        shapley_values = shapley_values.align_to(
            ..., NAME_FEATURES, NAME_META_CHANNELS)
        # get the norm of the vectors for each of the pixels, based on
        # which the pruning will be performed
        abs_values = torch.linalg.norm(
            shapley_values.rename(None), ord=1, dim=-1)
        # generate top-k
        top_k = torch.topk(
            abs_values.rename(None), k, largest=False
        )[0].max(1, keepdim=True)[0]  # threshold of the values to prune
        shapley_values = shapley_values * (
                abs_values > top_k).unsqueeze(-1)

        shapley_values = shapley_values.align_to(*names)

        return shapley_values


class DeepButterflyShapleyNetwork(DeepShapleyNetwork, ABC):
    r"""
    Butterfly mechanism from FFT to Deep {\sc ShapNet}
    """

    def __init__(
            self,
            lists_modules: List[List[ShapleyModule]] = None,
            reference_values: torch.Tensor = None,
            named_output: bool = False,
            residual: bool = True,
            pruning: List[float] or float = 0.,
            pruning_dims: List[TensorName] = None,
            utilized_dims: Tuple[TensorName] = (
                    NAME_FEATURES, NAME_META_CHANNELS)
    ):
        r"""

        Args:
            input channels and output channels
            lists_modules (): The modules to put inside the Shallow {\sc ShapNet}
                        components,
            reference_values (): the reference values for each input feature
            named_output (): if the output will be named
        """
        lists_keys = self._generate_keys(lists_modules)
        list_shapnets = [
            ShallowShapleyNetwork(
                module_dict={key: module for key, module in zip(keys, modules)}
            ) for keys, modules in zip(lists_keys, lists_modules)
        ]
        super(DeepButterflyShapleyNetwork, self).__init__(
            list_shapnets=list_shapnets,
            reference_values=reference_values,
            named_output=named_output,
            residual=residual,
            pruning=pruning,
            pruning_dims=pruning_dims,
            utilized_dims=utilized_dims
        )

    @staticmethod
    def _generate_keys(list_shapnets) -> List[List[str]]:
        r"""
        Generate a list of keys for Butterfly mechanism

        Returns:
            keys for different modules in different stages

        """
        features = list_shapnets[0].dimensions.features
        num_stages = int(np.ceil(np.log2(features)))
        num_module = 2 ** (num_stages - 1)
        # prepare the keys
        list_keys = [[str(get_indices(
            total_num=num_stages, index_of_interest=i, index_in_index=j
        )) for i in range(num_module)] for j in range(num_stages)]
        return list_keys
