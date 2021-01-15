#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
The Module contains base class for ShapNets and Shapley Modules
"""
# =============================================================================
# Imports
# =============================================================================
import re
from abc import ABC
from typing import List, Tuple, TypeVar

import torch
import torch.nn as nn
from scipy.special import factorial

from ShapNet.utils import ListModuleDimensions, ModuleDimensions, \
    NAME_BATCH_SIZE, \
    NAME_FEATURES, NAME_FEATURES_META_CHANNEL, NAME_META_CHANNELS, \
    NAME_NUM_PASSES, generate_binary_sequence, get_indices, \
    named_tensor_add_dim, named_tensor_vector_indexing_single_dim

T = TypeVar("T")


# =============================================================================
# Classes
# =============================================================================
class ShapleyNetwork(nn.Module, ABC):
    r"""
    This is the base class for every class that we use later to build {\sc
    ShapNet}
    
    Attributes:
        named_output: if the output should be named using the named tensor
        dimensions: the dimensions object which makes keeping track of the
            dimensions of the representations easier.
        reference_values: the reference values for the modules, Shallow 
            ShapNet, and Deep ShapNet
    """
    named_output: bool
    dimensions: ModuleDimensions or ListModuleDimensions
    reference_values: torch.Tensor

    def __init__(
            self,
            dimensions: ModuleDimensions or ListModuleDimensions = None,
            reference_values: torch.Tensor = None,
            named_output: bool = False,
    ) -> None:
        r"""
        The initialization method for the base class
        Args:
            dimensions (): ModuleDimensions that contains the features,
                                           in_channels,
                                       and output_channels
            reference_values (): the reference values for the input dimensions
                    Default: torch.zeros()
            named_output (): if we want our output to be named tensor
        """
        super(ShapleyNetwork, self).__init__()
        # Assign the values
        self.named_output = named_output
        self.dimensions = dimensions

        # assign reference values
        # default to zero if not provided
        self.reference_values = nn.Parameter(
            reference_values or torch.zeros(
                self.dimensions.features, self.dimensions.in_channel,
                requires_grad=False
            ), requires_grad=False,
        ).rename(NAME_FEATURES, NAME_META_CHANNELS)

    def forward(
            self,
            inputs: torch.Tensor,
            explain: bool = False,
            dim_channels: int = None,
            named_output: bool = False
    ) -> torch.Tensor or Tuple[torch.Tensor]:
        r"""
        The usual forward method in PyTorch
        We structure the method s.t. one can write explained forward and
        unexplained version independently when possible. That being said,
        this unexplained version has a generic implementation where it sums
        over the input feature importance, which is the local accuracy
        property of Shapley values
        Args:
            inputs (): input tensor
            explain (): if the explanation is wanted
            dim_channels (): the dimension of input channels, this is
                not generally used, but put here for flexibility
            named_output (): if we want the output to be named

        Returns:
            if explain:
                Will return the explanations, in general, in the shape of (
                batch_size, features, num_classes)
            if not explain:
                Will return the output of the underlying model

        """
        # get the number of channel dimensions
        dim_channels = dim_channels or self.dimensions.in_channel
        # one of the operations is to prepare the values, which may change
        # for different model classes defined afterwards, and be overloaded
        inputs = self._prepare_input_dim(inputs, dim_channels, explain=explain)
        # depending on variable {explain}, we pass the input into two
        # different forward methods
        if explain:
            output = self.explained_forward(inputs)  # if we want to explain
        else:
            output = self.unexplained_forward(inputs)  # if not

        # if we want the output to be named or not
        named_output = named_output or self.named_output
        if named_output:
            return output

        # if the output is torch tensor or just list
        if not isinstance(output, torch.Tensor):
            return [out.rename(None) for out in output]
        return output.rename(None)

    def explained_forward(self, inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, ...] or torch.Tensor:
        r"""
        To be overloaded

        The explained forward computation
        Args:
            inputs (): the prepared inputs

        Returns: Shapley values
            of shape (batch_size, features, output_channels)
        """
        raise NotImplementedError("This should be overloaded")

    def unexplained_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        A generic implementation where the explanation is summed over the
        input features
        Args:
            inputs (): the prepared inputs

        Returns:
            output of the underlying model
            of shape (batch_size, output_channels)
        """
        return self.explained_forward(inputs).sum([NAME_FEATURES])

    def _prepare_input_dim(
            self,
            inputs: torch.Tensor,
            in_channels: int = 1,
            explain: bool = False
    ) -> torch.Tensor:
        r"""
        Since every model architecture that we write in the
        {\sc ShapNet} module takes input of the shape
        (*, features, in_channels),  we need to prepare the data from
        (*, features)
        Args:
            inputs (): of shape (*, features)
            in_channels: the number of channels to group to
                an input feature

        Returns:
            of shape (*, features, 1)
        """
        batch_shape = list(inputs.shape[:-1])  # (*)
        # first add the shapes for x
        inputs = inputs.rename(None).reshape(
            # (*, features, 1)
            *(batch_shape + [inputs.shape[-1] // in_channels, in_channels])
        )
        # now we rename the tensor
        return inputs.refine_names(
            NAME_BATCH_SIZE, ..., NAME_FEATURES, NAME_META_CHANNELS
        )

    @staticmethod
    def _get_indices_from_key(item) -> List[int]:
        r"""
        generate list of indices from a string of the form "(1,2,3)"
        Args:
            item (): a string of the form "(i, j, k)"

        Returns:
            list of integer indices `[i, j, k]`
        """
        return [int(i) for i in re.findall(r"\d+", item)]

    def to(self, *args, **kwargs):
        r"""
        This is only needed because of the support, or lack thereof, of the 
        named tensor experimental feature.

        Override the .to method of the module as the older implementation
        does not work with named tensors even if tensor names are removed.
        Args:
            *args (): follows the .to method overloaded
            **kwargs (): follows the .to method overloaded

        Returns:
            the {\sc ShapNet} on the designated device
        """
        super(ShapleyNetwork, self).to(*args, **kwargs)  # Do the original .to
        self._recursive_named_tensor_to(*args, **kwargs)  # work on the named
        return self

    def _recursive_named_tensor_to(self, *args, **kwargs) -> None:
        r"""
        This method performs before or after the .to method from the base
        class nn.Module
        """

        # first we define the inner recursive function
        def __recursive(module: nn.Module):
            """
            Recursively perform .to operation on named tensors
            """
            if hasattr(module, "masks"):
                module.masks = module.masks.to(
                    *args, **kwargs)
            if hasattr(module, "reference_values"):
                module.reference_values = module.reference_values.to(
                    *args, **kwargs)
            if hasattr(module, "subtraction_matrix"):
                module.subtraction_matrix = module.subtraction_matrix.to(
                    *args, **kwargs)
            # Do the recursive part
            for child in module.children():
                __recursive(child)

        # now call the function
        __recursive(self)


class ShapleyModule(ShapleyNetwork, ABC):
    r"""
    This is the general class for the Shapley module described in the paper.
    The goal of such module is to curb the computation by sparsity.

    Instance Attributes:
        inner_function: the underlying model f
        passes: the number of passes used to compute the Shapley values
        masks: the binary mask to generate different samples to form absence
        weights: the weights for computing the summation
    """

    def __init__(
            self,
            inner_function: nn.Module,
            dimensions: ModuleDimensions = None,
            reference_values: torch.Tensor = None,
    ) -> None:
        r"""
        Initialization function.

        Args:
            inner_function (): the model inside the Shapley Module,
                               the  underlying model f
            dimensions (): the ModuleDimensions that contains the dimensions
            reference_values (): the reference values for each of
                                 the input features
                of shape (features)
        """
        super(ShapleyModule, self).__init__(
            dimensions=dimensions,
            reference_values=reference_values,
        )
        self.inner_function = inner_function  # assign the inner function
        # The number of forward passes to compute Shapley values
        self.passes = 2 ** self.dimensions.features
        # used to mix the reference values with the actual values
        self.masks = generate_binary_sequence(dimensions.features)
        self.masks = self.masks.refine_names(
            NAME_NUM_PASSES, NAME_FEATURES, NAME_META_CHANNELS
        )  # name the tensor
        # compute the weights for the computation of marginal difference
        self.subtraction_matrix = self._get_subtraction_matrix()

    @staticmethod
    def _preprocess(inputs: torch.Tensor) -> torch.Tensor:
        r"""
        This is the default behavior of the model which regards 

        Args:
            inputs (): add dimension in named tensor

        Returns:
            data prepared for further processing

        """
        return named_tensor_add_dim(inputs, 0, NAME_NUM_PASSES)

    def explained_forward(
            self,
            inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward to explain the inner function

        This function will expand the input to create samples that are
        interpolating between the input and the reference values, which will
        then be passed into the model in batches, as a way to mitigate the
        computational overhead and at the cost of memory.

        Args:
            inputs (): the input vectors, of shape (*, features,
            in_channels)

        Returns:
            bias, Shapley values,
            bias of shape
                (batch_size, *, output_channels)
            Shapley values of shape
                (batch_size, *, features, output_channels)
        """
        # We first extend the input
        inputs_reshaped = self._preprocess(inputs)
        # broadcast the masks
        masks = self.masks.align_as(inputs_reshaped)
        # mix the reference values and the actual input
        inputs_extended = masks * inputs_reshaped + (
                1 - masks) * self.reference_values
        inputs_reshaped = inputs_extended.flatten(
            [NAME_FEATURES, NAME_META_CHANNELS], NAME_FEATURES_META_CHANNEL
        )
        # keep current names
        names = inputs_reshaped.names
        # Pass into the model in batches
        function_outputs = self.inner_function(
            inputs_reshaped.rename(None)
        ).refine_names(*names)
        function_outputs = function_outputs.rename(
            feature_times_channels=NAME_META_CHANNELS
        )  # (2 ** features, batch_size, *, output_channels)
        # Compute the Shapley values
        shapley_values = self.compute_shapley(function_outputs)
        shapley_values = shapley_values.align_to(
            NAME_BATCH_SIZE, ..., NAME_META_CHANNELS)
        # return the bias and the shapley values
        return function_outputs[-1], shapley_values

    def unexplained_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        run the inner function
        Args:
            inputs (): the input vectors, of shape (*, features,
            in_channels)

        Returns:
            model output of shape (batch_size, num_classes)
        """
        return self.inner_function(inputs.flatten([
            NAME_FEATURES, NAME_META_CHANNELS], NAME_FEATURES_META_CHANNEL), )

    def compute_shapley(self, function_outputs: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            function_outputs (): of shape (2 ** features, batch_size,
            output_channels)

        Returns:
            Shapley values for each of the variables
                should be of shape (batch_size, self.m, output_channels)
        """
        shapley_values = torch.matmul(
            function_outputs.align_to(..., NAME_NUM_PASSES),
            self.subtraction_matrix
        ).align_to(NAME_BATCH_SIZE, ..., NAME_FEATURES, NAME_META_CHANNELS)

        return shapley_values

    def _get_subtraction_matrix(self) -> torch.Tensor:
        """
        This generates a matrix, with which the multiplication of the
        function output, will return the Shapley representation

        function_outputs is of shape (2 ** features, batch_size, *,
            output_channels), hence we need the output tensor to be of shape
            (features, 2 ** features), the multiplication of which will
            result in a representation of shape
            (features, batch_size, *, output_channels).

        Returns:
            The matrix that perform the subtraction and summation.

        """
        shapley_weights = self._weight_compute()
        num_features = self.dimensions.features
        matrix = torch.zeros(2 ** num_features, num_features)
        for i in range(num_features):
            minuend, subtrahend = self._generate_subtraction_sequence(i)
            weights = named_tensor_vector_indexing_single_dim(
                shapley_weights, NAME_NUM_PASSES, indices=subtrahend
            ).rename(None)
            matrix[minuend, i] = weights
            matrix[subtrahend, i] = - weights

        matrix = matrix.refine_names(NAME_NUM_PASSES, NAME_FEATURES)

        return matrix

    def _generate_subtraction_sequence(self, index: int) -> torch.LongTensor:
        r"""
        To generate the sequence of subtraction for the current index

        Args:
            index (): the index of interest

        Returns:
            tensor of size (2, 2 ** (self.m - 1))
        """
        self.passes = 2 ** self.dimensions.features
        sequence = torch.empty(2, self.passes // 2).long()  # placeholder
        for i in range(self.passes // 2):
            sequence[0, i], sequence[1, i] = get_indices(
                total_num=self.passes, index_of_interest=index,
                index_in_index=i
            )

        return sequence

    def _weight_compute(self) -> torch.Tensor:
        r"""
        Compute the weights of the differences

        Returns:
            weights to compute Shapley values with
        """
        z_size = self.masks.sum(-2).squeeze().numpy()
        m_factorial = factorial(self.dimensions.features)
        z_factorial = factorial(z_size)
        mz_factorial = factorial(
            self.dimensions.features - z_size - 1)
        weight = z_factorial * mz_factorial / m_factorial

        return torch.from_numpy(weight).float().refine_names(NAME_NUM_PASSES)

