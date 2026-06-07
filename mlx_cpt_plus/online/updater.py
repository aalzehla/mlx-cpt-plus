"""Online model updater."""

from typing import List, Optional
from mlx_cpt_plus.config import Config


class Updater:
    """Updates model with new data.
    
    Provides incremental learning capabilities for
    updating the model with new sequences.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize updater.
        
        Args:
            config: Configuration for updates.
        """
        self.config = config or Config()
        self._update_count = 0
    
    def update(self, sequences: List[List[int]]) -> int:
        """Update model with new sequences.
        
        Args:
            sequences: New sequences to learn.
            
        Returns:
            Number of sequences processed.
        """
        # Placeholder for actual update logic
        self._update_count += len(sequences)
        return len(sequences)
    
    def partial_fit(self, sequence: List[int]) -> None:
        """Perform partial fit on a single sequence.
        
        Args:
            sequence: Sequence to learn.
        """
        self.update([sequence])
    
    def get_update_count(self) -> int:
        """Get total number of updates.
        
        Returns:
            Number of updates performed.
        """
        return self._update_count
    
    def reset(self) -> None:
        """Reset the updater."""
        self._update_count = 0
    
    def __repr__(self) -> str:
        return f"Updater(updates={self._update_count})"