#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rui Wang
# =============================================================================
"""
Since named tensor indexing is not supported, I'll simply write one

Note that
    1. Slice index is supported so not here
"""

# =============================================================================
# Imports
# =============================================================================
from typing import List

import torch

from .private_methods import TensorName, add_dim, dummy, name_insert


# =============================================================================
# functions
# =============================================================================
def _named_tensor_generic_operation(
        tensor: torch.Tensor,
        tensor_ops_pre: callable = dummy,
        tensor_ops_post: callable = dummy,
        name_ops: callable = dummy) -> torch.Tensor:
    """
    generic base function used by others
    First store the names
    Args:
        tensor (): the named tensor to work on
        tensor_ops_pre (): the operation before the name is removed
        tensor_ops_post (): the operation after the name is removed that act on
        the tensor
        name_ops (): the operation to act on names

    Returns:

    """
    # Save the names in names_old and then remove the names from the tensor
    tensor = tensor_ops_pre(tensor)
    names_old = tensor.names
    tensor = tensor.rename(None)

    # operations
    names_new = name_ops(names_old)  # modify the names
    tensor = tensor_ops_post(tensor)  # change the tensor accordingly

    return tensor.refine_names(*names_new)  # name the tensor


def named_tensor_add_dim(
        tensor: torch.Tensor, index: int, name: TensorName) -> \
        torch.Tensor:
    """
    Used as a replacement of .unsqueeze method
    Args:
        tensor (): the tensor to add an dimension
        index (): the index of to add dimension in
        name (): the name of the new dimension

    Returns:
        named dimension added tensor
    """
    return _named_tensor_generic_operation(
        tensor,
        tensor_ops_post=lambda _tensor: add_dim(_tensor, index),
        name_ops=lambda _names: name_insert(_names, index, name)
    )


def named_tensor_vector_indexing_single_dim(
        tensor: torch.Tensor, dim: TensorName,
        indices: torch.LongTensor or list
):
    """
    This is implemented as a way to support named tensor indexing
    Args:
        tensor (): the named tensor to index
        dim (): the dimension
        indices ():

    Returns:

    """
    # # First a single element from a one-dimensional vector
    names = tensor.names
    index_dim = names.index(dim)
    tensor = tensor.rename(None)
    tensor = tensor.transpose(0, index_dim)
    tensor = tensor[indices]
    tensor = tensor.transpose(0, index_dim)
    return tensor.refine_names(*names)


def named_tensor_repeat(tensor: torch.Tensor, repeat_dims: list):
    """
    Repeat method in named tensor
    Args:
        tensor (): the tensor to repeat
        repeat_dims (): the times for repeat for each dimension

    Returns:
        repeated tensor
    """
    names = tensor.names
    tensor = tensor.rename(None)

    tensor = tensor.repeat(repeat_dims)

    return tensor.refine_names(*names)


def named_tensor_squeeze(tensor: torch.Tensor):
    """
    Not yet implemented
    Args:
        tensor ():

    Returns:

    """
    pass


def named_tensor_split(tensor: torch.Tensor, split_size: int, dim: str) \
        -> List[torch.Tensor]:
    """
    As working in torch.split but added list()
    Args:
        tensor (): the tensor to split
        split_size (): the size of the splits
        dim (): the dimension of each of the splits

    Returns:
        list of splits
    """
    dim_index = tensor.names.index(dim)
    return list(tensor.split(split_size, dim=dim_index))


def named_tensor_get_dim(
        tensor: torch.Tensor, name: TensorName or List[TensorName]
) -> List[int] or int:
    """

    Args:
        tensor (): get the number of dimension by name
        name ():

    Returns:

    """
    single = False
    if isinstance(name, TensorName):
        name = [name]
        single = True
    dicts = {tensor_name: val for tensor_name, val in
             zip(tensor.names, tensor.shape)}

    if single:
        return dicts[name[0]]
    else:
        return [dicts[tensor_name] for tensor_name in name]


# =============================================================================
# testing
# =============================================================================
if __name__ == "__main__":
    named_tensor = torch.randn(10, 10, names=("C", "G"))
    named_tensor_add_dim(named_tensor, -1, "E")
    named_tensor_vector_indexing_single_dim(
        named_tensor, "G", [torch.tensor([1, 2, 3])])
