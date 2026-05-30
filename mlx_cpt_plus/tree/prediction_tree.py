"""Prediction tree implementation."""

from typing import List, Optional, Dict
from mlx_cpt_plus.tree.node import Node
from mlx_cpt_plus.exceptions import TreeError


class PredictionTree:
    """Prediction tree for sequence modeling.
    
    The prediction tree stores sequences and enables efficient
    prediction of next items based on context.
    """
    
    def __init__(self, max_depth: int = 10, branching_factor: int = 16):
        """Initialize prediction tree.
        
        Args:
            max_depth: Maximum depth of the tree.
            branching_factor: Maximum children per node.
        """
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.root = Node(node_id=0)
        self._node_count = 1
        self._nodes: Dict[int, Node] = {0: self.root}
    
    def insert(self, sequence: List[int]) -> None:
        """Insert a sequence into the tree.
        
        Args:
            sequence: Sequence of items to insert.
        """
        current = self.root
        for item in sequence:
            if not current.has_child(item):
                if len(current.children) >= self.branching_factor:
                    # Tree is full at this level, skip
                    return
                if current.depth() >= self.max_depth:
                    # Max depth reached
                    return
                new_node = Node(node_id=self._node_count, parent=current)
                self._nodes[self._node_count] = new_node
                self._node_count += 1
                current.add_child(item, new_node)
            current = current.get_child(item)
    
    def get_next_candidates(self, context: List[int]) -> List[int]:
        """Get candidate items for next prediction.
        
        Args:
            context: Context items.
            
        Returns:
            List of candidate items.
        """
        node = self._find_node_for_context(context)
        if node is None:
            return []
        return list(node.children.keys())
    
    def _find_node_for_context(self, context: List[int]) -> Optional[Node]:
        """Find the node corresponding to a context.
        
        Args:
            context: Context items.
            
        Returns:
            Node if found, None otherwise.
        """
        current = self.root
        for item in context:
            child = current.get_child(item)
            if child is None:
                return None
            current = child
        return current
    
    def size(self) -> int:
        """Get number of nodes in the tree.
        
        Returns:
            Number of nodes.
        """
        return self._node_count
    
    def clear(self) -> None:
        """Clear the tree."""
        self.root = Node(node_id=0)
        self._node_count = 1
        self._nodes = {0: self.root}
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"PredictionTree(nodes={self.size()}, depth={self.max_depth})"