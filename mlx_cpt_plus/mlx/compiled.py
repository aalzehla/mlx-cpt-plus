"""Compiled MLX functions."""

import mlx.core as mx
from typing import Callable


def compiled(func: Callable) -> Callable:
    """Compile a function for faster execution.
    
    Args:
        func: Function to compile.
        
    Returns:
        Compiled function.
    """
    return mx.compile(func)