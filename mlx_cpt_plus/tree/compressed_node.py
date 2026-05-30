"""Compressed node for memory-efficient tree storage."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CompressedNode:
    """Memory-efficient compressed node representation.
    
    Uses compression techniques to reduce memory footprint
    of the prediction tree.
    """
    
    node_id: int
    child_indices: List[int]
    child_items: List[int]
    context_hash: int
    
    def get_child_index(self, item: int) -> Optional[int]:
        """Get child index for an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Child index if found, None otherwise.
        """
        try:
            pos = self.child_items.index(item)
            return self.child_indices[pos]
        except ValueError:
            return None
    
    def add_child(self, item: int, child_idx: int) -> None:
        """Add a child mapping.
        
        Args:
            item: Item for the child.
            child_idx: Index of the child node.
        """
        self.child_items.append(item)
        self.child_indices.append(child_idx)
    
    def decompress(self) -> 'Node':
        """Decompress to a full Node.
        
        Returns:
            Decompressed Node instance.
        """
        from mlx_cpt_plus.tree.node import Node
        node = Node(node_id=self.node_id)
        for item, idx in zip(self.child_items, self.child_indices):
            node.children[item] = Node(node_id=idx)
        return node
    
    def __repr__(self) -> str:
        return f"CompressedNode(id={self.node_id}, children={len(self.child_items)})"