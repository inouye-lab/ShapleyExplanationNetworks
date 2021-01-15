#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
This module contains the construction of Shallow ShapNets as described
in the paper with other variants used in the experiments
"""
# =============================================================================
# Imports
# =============================================================================
import math
import re
from abc import ABC
from typing import List, Tuple

import torch
from torch.nn import ModuleDict

from .basic import ShapleyModule, ShapleyNetwork
from .utils import ModuleDimensions, NAME_BATCH_SIZE, NAME_FEATURES, \
    NAME_META_CHANNELS, \
    NAME_NUM_PAIRS, flatten, named_tensor_vector_indexing_single_dim


# =============================================================================
# Classes
# =============================================================================
class ShallowShapleyNetwork(ShapleyNetwork, ABC):
    r"""
    The architecture of Shallow Shapley Networks.

    How it works:
        For each of the Shapley modules, we have a corresponding index list,
        which indicates which inputs to take into account in that Shapley
        module. From there we could just iterate through the dictionary
    """

    def __init__(
            self,
            module_dict: ModuleDict or dict,
            reference_values: torch.Tensor = None,
            dimensions: ModuleDimensions = None
    ):
        r"""
        Initialization

        Args:
            module_dict (): Only support the following form of ``modules``:
                ModuleDict[str:ShapleyModule]: this contains already
                initialized torch
                    modules, each of which will be wrapped in a
                    :class:`~{\sc ShapNet}.ShapleyModule`.
            reference_values (): the reference values for each of the variables
                Defaults to zeros
        """
        # input validation: correct the type of the module_dict argument
        module_dict = module_dict if not isinstance(module_dict, ModuleDict) \
            else ModuleDict(module_dict)
        module_dim = list(module_dict.values())[0].dimensions
        # input validation: correct the input for dimensions
        if dimensions is None:
            dimensions = ModuleDimensions(
                features=self._get_input_features(module_dict),
                in_channel=module_dim.in_channel,
                out_channel=module_dim.out_channel
            )

        super(ShallowShapleyNetwork, self).__init__(
            dimensions=dimensions,
            reference_values=reference_values,
        )
        if isinstance(module_dict, ModuleDict):
            self.module_dict = module_dict
        else:
            self.module_dict = ModuleDict(module_dict)
        # align the reference values of the entire model with those of the 
        # Shapley modules
        self.align_reference_values()

        assert len(self.module_dict) != 0
        assert isinstance(self.module_dict, ModuleDict)
        assert isinstance(list(module_dict.values())[0], ShapleyModule)

    def _get_input_features(self, module_dict: ModuleDict) -> int:
        r"""
        Get the number of input features by the ModuleDict input instance

        Args:
            module_dict (): the dictionary which contains the keys and 
                their corresponding modules.

        Returns:
            the number of features used.

        """
        indices = [self._get_keys(key) for key in module_dict.keys()]
        return len(set(flatten(indices)))

    def explained_forward(self, inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The forward method that performs explanation
        Args:
            inputs (): of shape (*, features, input_channels)

        Returns:
            the exact Shapley values for each input features

        """
        # setup place holders
        out = [0] * self.dimensions.features
        biases = 0
        for _, (key, module) in enumerate(self.module_dict.items()):
            indices = self._get_keys(key)
            bias, output = module.explained_forward(
                named_tensor_vector_indexing_single_dim(
                    inputs, dim=NAME_FEATURES, indices=indices))
            output = output.rename(None)

            biases = bias + biases
            for i, index in enumerate(indices):
                out[index] = out[index] + output[..., i, :]

        zeros = torch.zeros_like(output[..., 0, :])
        out = [o if isinstance(o, torch.Tensor) else zeros for o in out]
        shapley_values = torch.stack(
            out, -2
        ).refine_names(
            NAME_BATCH_SIZE, ..., NAME_FEATURES, NAME_META_CHANNELS)

        return shapley_values, biases

    def unexplained_forward(self, inputs: torch.Tensor):
        r"""
        The forward operation that is not explained, this is the usual
        forward method for the underlying function. In other words, this 
        works exactly as the underlying function does.

        Args:
            inputs: the input samples.
        """
        output = 0
        for _, (key, module) in enumerate(self.module_dict.items()):
            indices = self._get_keys(key)
            out = module.unexplained_forward(
                named_tensor_vector_indexing_single_dim(
                    inputs, dim=NAME_FEATURES, indices=indices))
            output = output + out

        return output

    @staticmethod
    def _get_keys(key_string: str) -> List[int]:
        r"""
        Get feature index from the key string
        :param key_string: the key string from the dictionary
        :return: the first feature index and the second feature index
        """
        return [int(key) for key in re.findall(r"\d+", key_string)]

    def align_reference_values(self, reference_values: torch.Tensor = None):
        """
        put the overall reference values to the first stage-reference values
        Returns:

        """
        if reference_values is None:
            reference_values = self.reference_values
        for key, module in self.module_dict.items():
            module.reference_values = named_tensor_vector_indexing_single_dim(
                reference_values, NAME_FEATURES,
                self._get_indices_from_key(key))


# ============================================================================
# now we generalize a bit to generalized additive models and overlapping 
# Shallow ShapNets
# ============================================================================
class ShallowSharedShapleyNetwork(ShallowShapleyNetwork, ABC):
    """
    This is the special instance which shares the computation,
    similar to that of the convolutional layers.
    """

    def __init__(
            self,
            dimensions: ModuleDimensions,
            shapley_module: ShapleyModule,
            reference_values: torch.Tensor = None,
    ):
        module_dict = {
            str(set([i for i in range(
                shapley_module.dimensions.features)])): shapley_module
        }
        super(ShallowSharedShapleyNetwork, self).__init__(
            module_dict=module_dict,
            reference_values=reference_values,
            dimensions=dimensions
        )
        self.shapley_module = shapley_module

    def explained_forward(
            self,
            inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        This method performs a shared operation across different groups/pairs
        of the input features.

        Notes:
            There are 3 sub-stages in this method:
            1. reshape: reshape the input s.t. it is ready for inner function
            2. module feed: feed the reshaped input into the module
            3. undo 1.

            Each of the stages can be overloaded to give flexibility

        Args:
            inputs ():

        Returns:
            The shapley values from the current shallow layer (stage)

        """
        # change the order in which the features are
        # arranged s.t. it follows certain mechanism
        # Reshaping to prepare to feed into the model
        inputs = self.reshape(inputs)
        # (batch_size, num_dim // 2, 2 * input_channels)

        # each of these: (batch_size, input_dim // 2, output_channels)
        # probably needs more permuting dimensions
        biases, inputs = self._feed_module(inputs)
        inputs = self.undo_reshape(inputs)
        # rearrange values

        return inputs, biases

    def reshape(self, shapley_values: torch.Tensor) -> torch.Tensor:
        r"""
        Reshaping method of this model
        Args:
            shapley_values (): the shapley values
                                (batch_size, features,
                                input_channels)

        Returns:

        """
        dim_input_features = shapley_values.shape[1]
        return shapley_values.unflatten(
            NAME_FEATURES,
            ((NAME_NUM_PAIRS, dim_input_features // 2), (NAME_FEATURES, 2))
        )

    def undo_reshape(self, shapley_values: torch.Tensor) -> torch.Tensor:
        r"""
        As the name suggested

        Args:
            shapley_values ():

        Returns:

        """
        return shapley_values.flatten(
            [NAME_NUM_PAIRS, NAME_FEATURES], NAME_FEATURES
        )

    def _feed_module(
            self, shapley_values_reshaped: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        r"""
        Feed the input into the module

        Notes:
            This is organized to promote flexibility

        Args:
            shapley_values_reshaped (): the reshaped Shapley values

        Returns:

        """
        return self.shapley_module.explained_forward(shapley_values_reshaped)


class GeneralizedAdditiveModel(ShallowShapleyNetwork, ABC):
    r"""
    Generalized Additive Model which is a special case of Shallow {\sc ShapNet}
    """

    def __init__(
            self,
            list_modules: List[ShapleyModule] = None,
            reference_values: torch.Tensor = None,
    ):
        r"""

        Args:
            list_modules ():
            reference_values ():
        """
        keys = [f"({i})" for i in range(len(list_modules))]
        module_dict = {key: module for key, module in zip(keys, list_modules)}

        super(GeneralizedAdditiveModel, self).__init__(
            module_dict=module_dict,
            reference_values=reference_values,
        )


class OverlappingShallowShapleyNetwork(ShallowShapleyNetwork, ABC):
    r"""
    As discussed in paper, where all of the features interact pair-wisely
    """

    def __init__(
            self,
            list_modules: List[ShapleyModule],
            reference_values: torch.Tensor = None,
    ):
        r"""
        This init differs in the original one in that it initialize
            a key list

        Args: See Also: :class:`~ShapleyNetwork`
        """
        # Initialize a key list
        features = int((1 + math.sqrt(len(list_modules))) // 2)
        keys = [
            [(i, j) for i in range(features) if i < j
             ] for j in range(features)]
        keys = [str(item) for sublist in keys for item in sublist]

        module_dict = {key: module for key, module in zip(keys, list_modules)}
        super(OverlappingShallowShapleyNetwork, self).__init__(
            module_dict=module_dict,
            reference_values=reference_values,
        )
