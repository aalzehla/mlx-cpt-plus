"""Configuration for MLX CPT+."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for CPT+ model.
    
    Attributes:
        max_sequence_length: Maximum length of input sequences.
        min_frequency: Minimum frequency for items to be included in vocabulary.
        max_vocabulary_size: Maximum vocabulary size.
        num_leaves: Number of leaves in the prediction tree.
        branching_factor: Branching factor for the tree.
        max_depth: Maximum depth of the prediction tree.
        learning_rate: Learning rate for online updates.
        use_mlx: Whether to use MLX acceleration.
        device: Device to use ('cpu' or 'gpu').
    """
    
    max_sequence_length: int = 1000
    min_frequency: int = 2
    max_vocabulary_size: int = 1_000_000
    num_leaves: int = 1000
    branching_factor: int = 16
    max_depth: int = 10
    learning_rate: float = 0.1
    use_mlx: bool = True
    device: str = "mps"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if self.max_vocabulary_size <= 0:
            raise ValueError("max_vocabulary_size must be positive")
        if self.num_leaves <= 0:
            raise ValueError("num_leaves must be positive")
        if self.branching_factor <= 0:
            raise ValueError("branching_factor must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.device not in ("cpu", "gpu","mps"):
            raise ValueError("device must be 'cpu', mps, or 'gpu'")