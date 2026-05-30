"""Tree builder for constructing prediction trees."""

from typing import List, Iterable
from mlx_cpt_plus.tree.prediction_tree import PredictionTree
from mlx_cpt_plus.exceptions import TreeError


class TreeBuilder:
    """Builds prediction trees from sequences.
    
    Provides methods for constructing and populating
    prediction trees from training data.
    """
    
    def __init__(self, max_depth: int = 10, branching_factor: int = 16):
        """Initialize tree builder.
        
        Args:
            max_depth: Maximum depth of the tree.
            branching_factor: Maximum children per node.
        """
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    
    def build(self, sequences: Iterable[List[int]]) -> PredictionTree:
        """Build a prediction tree from sequences.
        
        Args:
            sequences: Iterable of sequences to insert.
            
        Returns:
            Built prediction tree.
        """
        tree = PredictionTree(
            max_depth=self.max_depth,
            branching_factor=self.branching_factor
        )
        for seq in sequences:
            tree.insert(seq)
        return tree
    
    def build_from_sequences(self, sequences: List[List[int]]) -> PredictionTree:
        """Build tree from a list of sequences.
        
        Args:
            sequences: List of sequences.
            
        Returns:
            Built prediction tree.
        """
        return self.build(sequences)
    
    def __repr__(self) -> str:
        return f"TreeBuilder(depth={self.max_depth}, branching={self.branching_factor})"