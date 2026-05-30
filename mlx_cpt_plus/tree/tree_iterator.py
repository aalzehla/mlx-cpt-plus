"""Tree iterator for traversing prediction trees."""

from typing import Iterator, Optional
from mlx_cpt_plus.tree.node import Node
from mlx_cpt_plus.tree.prediction_tree import PredictionTree


class TreeIterator:
    """Iterator for traversing prediction trees.
    
    Provides depth-first and breadth-first traversal
    of prediction tree nodes.
    """
    
    def __init__(self, tree: PredictionTree):
        """Initialize tree iterator.
        
        Args:
            tree: Tree to iterate over.
        """
        self.tree = tree
    
    def depth_first(self) -> Iterator[Node]:
        """Iterate over tree in depth-first order.
        
        Yields:
            Nodes in depth-first order.
        """
        yield self.tree.root
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                yield child
                stack.append(child)
    
    def breadth_first(self) -> Iterator[Node]:
        """Iterate over tree in breadth-first order.
        
        Yields:
            Nodes in breadth-first order.
        """
        queue = [self.tree.root]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children.values())
    
    def get_paths(self) -> Iterator[List[int]]:
        """Get all paths from root to leaves.
        
        Yields:
            Paths as lists of items.
        """
        def _get_path(node: Node, current_path: List[int]):
            if not node.children:
                yield current_path
            for item, child in node.children.items():
                yield from _get_path(child, current_path + [item])
        
        yield from _get_path(self.tree.root, [])
    
    def __repr__(self) -> str:
        return f"TreeIterator(tree={self.tree})"