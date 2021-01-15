#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
This module contains the construction of ShapNets as described
in the paper with other variants used in the experiments

These classes are only for vision tasks
"""

# =============================================================================
# Imports
# =============================================================================
from abc import ABC
from typing import List, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import ShapleyModule
from .deep import DeepShapleyNetwork
from .shallow import ShallowShapleyNetwork, ShallowSharedShapleyNetwork
from .utils import ModuleDimensions, NAME_BATCH_SIZE, \
    NAME_FEATURES, NAME_FEATURES_META_CHANNEL, NAME_META_CHANNELS, \
    NAME_PATCHES, named_tensor_get_dim
from .utils.named_tensor_utils import NAME_HEIGHT, NAME_WIDTH

T = TypeVar("T")


# =============================================================================
# Classes
# =============================================================================
class ShallowConvShapleyNetwork(ShallowSharedShapleyNetwork, ABC):
    r"""
    This is as described in the paper under paragraph 
    \paragraph{Deep {\sc ShapNet} for Images}

    We simply swap the matrix multiplication in convolutional layers, and
    substitute in the Shapley Modules

    This is implemented by utilizing PyTorch's nn.Unfold and nn.Fold modules

    """

    class Folder(nn.Module, ABC):
        """
        performs folding and unfolding
        """
        height: int
        width: int

        def __init__(
                self,
                kernel_size: Union[T, Tuple[T, ...]],
                dilation: Union[T, Tuple[T, ...]] = 1,
                padding: Union[T, Tuple[T, ...]] = 0,
                stride: Union[T, Tuple[T, ...]] = 1,
        ):
            r"""

            Args:
                The following are the same as `nn.Conv2d`
                kernel_size ():
                dilation ():
                padding ():
                stride ():

                The following follows the other {\sc ShapNet} models
            """
            super().__init__()
            self.fold_params = dict(
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride
            )
            self.unfold = nn.Unfold(**self.fold_params)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            r"""

            Args:
                inputs (): the usual image representation

            Returns:
                The representation following the output of nn.Unfold
                (batch_size, kernel_size * channels, patches)

            """
            self.height, self.width = inputs.shape[-2], inputs.shape[-1]
            return self.unfold(inputs.rename(None)).refine_names(
                NAME_BATCH_SIZE, NAME_FEATURES_META_CHANNEL, NAME_PATCHES)

        def fold(self, inputs: torch.Tensor, height: int,
                 width: int) -> torch.Tensor:
            r"""
            Same as above

            Args:
                inputs ():
                height ():
                width ():

            Returns:

            """
            return F.fold(inputs.rename(None), output_size=(
                height, width
            ), **self.fold_params).refine_names(
                NAME_BATCH_SIZE, NAME_META_CHANNELS, NAME_HEIGHT, NAME_WIDTH)

    def __init__(
            self,
            shapley_module: ShapleyModule,
            reference_values=None,
            kernel_size: Union[T, Tuple[T, ...]] = None,
            dilation: Union[T, Tuple[T, ...]] = 1,
            padding: Union[T, Tuple[T, ...]] = 0,
            stride: Union[T, Tuple[T, ...]] = 1,
    ):
        r"""
        The instantiation function

        The arguments including kernel_size, dilation, padding and stride
        are following the convention of nn.Conv2d
        Args:
            shapley_module (): the Shapley Module with which the matrix 
                multiplication in convolution is replaced.
            reference_values (): the reference values for the batch
            kernel_size ():
            dilation ():
            padding ():
            stride ():

        """
        kernel_size = [kernel_size, kernel_size] \
            if isinstance(kernel_size, int) else kernel_size
        dimensions = ModuleDimensions(
            features=int(np.prod(kernel_size)),
            in_channel=shapley_module.dimensions.in_channel,
            out_channel=shapley_module.dimensions.out_channel
        )
        super(ShallowConvShapleyNetwork, self).__init__(
            dimensions=dimensions, shapley_module=shapley_module,
            reference_values=reference_values
        )
        self.folder = self.Folder(
            kernel_size=kernel_size, dilation=dilation, padding=padding,
            stride=stride, )

        self.kernel_size, self.dilation = kernel_size, dilation
        self.padding, self.stride = padding, stride

    def reshape(self, shapley_values: torch.Tensor) -> torch.Tensor:
        r"""
        In this instantiation, this method tries to unfold the representation
        and prepare for the Shapley Module. 

        Args:
            shapley_values (): the input Shapley representation

        Returns:
            the prepared Shapley representation for the Shapley module that 
            follows

        """
        # Prepare the input, at the end should be of shape
        shapley_values_un = self.folder(shapley_values)
        # (batch_size, patches, features, meta-channels)
        shapley_values = shapley_values_un.unflatten(
            NAME_FEATURES_META_CHANNEL, [
                (NAME_META_CHANNELS, self.dimensions.in_channel),
                (NAME_FEATURES, np.prod(self.kernel_size)),
            ]
        )
        shapley_values = shapley_values.align_to(
            ..., NAME_FEATURES, NAME_META_CHANNELS)
        return shapley_values

    def undo_reshape(self, shapley_values: torch.Tensor) -> torch.Tensor:
        r"""
        This folds the representation back into the canonical image 
        representation


        Args:
            shapley_values (): the representation after the Shapley Module

        Returns:
            The image representation

        """
        # prepare the output for folding, at the end should be
        # (batch_size, features_times_channels, patches)
        shapley_values = shapley_values.align_to(
            ..., NAME_META_CHANNELS, NAME_FEATURES)
        shapley_values = shapley_values.flatten(
            [NAME_META_CHANNELS, NAME_FEATURES],
            NAME_FEATURES_META_CHANNEL)
        shapley_values = shapley_values.align_to(
            ..., NAME_FEATURES_META_CHANNEL, NAME_PATCHES)

        # the folding operation
        shapley_values = self.folder.fold(
            shapley_values, self.folder.height, self.folder.width)

        return shapley_values


class DeepConvShapNet(DeepShapleyNetwork, ABC):
    r"""
    The Deep {\sc ShapNet}} for images

    """

    def __init__(
            self,
            list_shapnets: List[ShallowConvShapleyNetwork],
            reference_values: torch.Tensor = None,
            residual: bool = True,
            named_output: bool = False,
            pruning: List[float] or float = 0.,
    ):
        r"""
        Instantiation
            
        Args:
            list_shapnets (): a list of Shallow ShapNet (with convolution)
            reference_values (): the reference values for the input
            residual (): if use residual connection
            named_output (): if to name the output
        """
        super(DeepConvShapNet, self).__init__(
            reference_values=reference_values,
            list_shapnets=list_shapnets,
            named_output=named_output,
            residual=residual,
            pruning=pruning,
        )

    def _prepare_input_dim(
            self,
            inputs: torch.Tensor,
            in_channels: int = 1,
            explain: bool = False
    ) -> torch.Tensor:
        r"""
        Do nothing, this is more of a book-keeping method.

        Args:
            inputs ():
            in_channels ():
            explain ():

        Returns:

        """
        return inputs

    def _in_stage_forward(
            self,
            id_stage: int,
            stage_layer: ShallowShapleyNetwork,
            shapley_previous: torch.Tensor,
            batch_size: int,
            dim_input_channels: int
    ) -> torch.Tensor:
        r"""
        The forward process used in each stage, note the split is for
        computational reasons like those in the paper real NVP and GLOW.

        Args:
            id_stage (): the index of the stage
            stage_layer (): the Shallow ShapNet layer
            shapley_previous (): the previous representation
            batch_size (): the batch size of the representation
            dim_input_channels (): the dimension of the input channels

        Returns:
            The Shapley representation

        """
        shapley_values_split = list(
            shapley_previous.split(stage_layer.dimensions.in_channel, dim=1))

        shapley_values_split[0] = super(
            DeepConvShapNet, self)._in_stage_forward(
            id_stage=id_stage,
            stage_layer=stage_layer,
            shapley_previous=shapley_values_split[0],
            batch_size=batch_size,
            dim_input_channels=dim_input_channels
        )

        return torch.cat(shapley_values_split, 1)

    def unexplained_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            inputs ():

        Returns:

        """
        return self.explained_forward(inputs).sum(1).sum(1)

    def _final_process(
            self, shapley_values: torch.Tensor, id_stage: int,
            *args, **kwargs) -> torch.Tensor:
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
        return shapley_values.align_to(
            NAME_BATCH_SIZE, ..., NAME_META_CHANNELS)

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
        num_pixels = named_tensor_get_dim(
            shapley_values, [NAME_HEIGHT, NAME_WIDTH])
        num_pixels = num_pixels[0] * num_pixels[1]
        k = int(num_pixels * pruning)  # the # to prune
        if pruning * k == 0:
            return shapley_values

        name_size = {name: size for name, size in
                     zip(shapley_values.names, shapley_values.shape)}

        shapley_values = shapley_values.align_to(
            ..., NAME_HEIGHT, NAME_WIDTH, NAME_META_CHANNELS).flatten(
            [NAME_HEIGHT, NAME_WIDTH], NAME_FEATURES)
        # get the norm of the vectors for each of the pixels, based on
        # which the pruning will be performed
        abs_values = torch.linalg.norm(
            shapley_values.rename(None), ord=1, dim=-1)
        # generate top-k
        top_k = torch.topk(
            abs_values.rename(None), k, largest=False
        )[0].max(1, keepdim=True)[0]  # threshold of the values to prune
        abs_values = abs_values.rename(NAME_BATCH_SIZE, NAME_FEATURES) > top_k
        shapley_values = shapley_values * abs_values.align_to(...,
                                                              NAME_META_CHANNELS)
        shapley_values = shapley_values.unflatten(
            NAME_FEATURES, [[NAME_HEIGHT, name_size[NAME_HEIGHT]],
                            [NAME_WIDTH, name_size[NAME_WIDTH]]])

        shapley_values = shapley_values.align_to(*name_size.keys())

        return shapley_values
