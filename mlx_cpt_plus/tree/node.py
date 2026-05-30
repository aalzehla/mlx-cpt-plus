"""Tree node implementation."""

from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class Node:
    """Node in the prediction tree.
    
    Each node represents a state in the prediction model, with
    transitions to child nodes based on input items.
    """
    
    node_id: int
    parent: Optional['Node'] = None
    children: Dict[int, 'Node'] = field(default_factory=dict)
    context: List[int] = field(default_factory=list)
    
    def add_child(self, item: int, child: 'Node') -> None:
        """Add a child node.
        
        Args:
            item: Item that leads to the child.
            child: Child node.
        """
        self.children[item] = child
        child.parent = self
    
    def get_child(self, item: int) -> Optional['Node']:
        """Get child node for an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Child node if found, None otherwise.
        """
        return self.children.get(item)
    
    def has_child(self, item: int) -> bool:
        """Check if node has a child for an item.
        
        Args:
            item: Item to check.
            
        Returns:
            True if child exists, False otherwise.
        """
        return item in self.children
    
    def get_path_to_root(self) -> List['Node']:
        """Get path from this node to root.
        
        Returns:
            List of nodes from this to root.
        """
        path = [self]
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def depth(self) -> int:
        """Get depth of this node.
        
        Returns:
            Depth from root (root is 0).
        """
        return len(self.get_path_to_root()) - 1
    
    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, depth={self.depth()}, children={len(self.children)})"