"""Partial fit for online learning."""

from typing import List, Optional
from mlx_cpt_plus.online.updater import Updater


class PartialFit:
    """Incremental learning interface.
    
    Provides a scikit-learn style partial_fit interface
    for online learning.
    """
    
    def __init__(self, updater: Optional[Updater] = None):
        """Initialize partial fit.
        
        Args:
            updater: Updater instance to use.
        """
        self.updater = updater or Updater()
    
    def partial_fit(self, X: List[List[int]], y: Optional[List[int]] = None) -> "PartialFit":
        """Incrementally fit the model.
        
        Args:
            X: Training sequences.
            y: Optional target (not used in unsupervised).
            
        Returns:
            self
        """
        self.updater.update(X)
        return self
    
    def learn_one(self, sequence: List[int]) -> None:
        """Learn from a single sequence.
        
        Args:
            sequence: Sequence to learn.
        """
        self.updater.partial_fit(sequence)
    
    def __repr__(self) -> str:
        return f"PartialFit(updater={self.updater})"