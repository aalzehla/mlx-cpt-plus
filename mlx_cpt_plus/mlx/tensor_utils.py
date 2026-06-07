"""MLX tensor utilities."""

import mlx.core as mx
from typing import List


def tensor_utils():
    """Return tensor utility functions."""
    return {
        "to_tensor": to_tensor,
        "from_tensor": from_tensor,
        "pad_sequence": pad_sequence,
    }


def to_tensor(data: List[int]) -> mx.array:
    """Convert list to MLX tensor.
    
    Args:
        data: List of integers.
        
    Returns:
        MLX array.
    """
    return mx.array(data, dtype=mx.int32)


def from_tensor(tensor: mx.array) -> List[int]:
    """Convert MLX tensor to list.
    
    Args:
        tensor: MLX array.
        
    Returns:
        List of integers.
    """
    return [int(x) for x in tensor.tolist()]


def pad_sequence(seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    """Pad a sequence to max length.
    
    Args:
        seq: Sequence to pad.
        max_len: Target length.
        pad_value: Padding value.
        
    Returns:
        Padded sequence.
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))