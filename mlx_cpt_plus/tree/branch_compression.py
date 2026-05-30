"""Branch compression for memory optimization."""

from typing import List, Tuple, Optional
from mlx_cpt_plus.tree.node import Node
from mlx_cpt_plus.tree.prediction_tree import PredictionTree


class BranchCompression:
    """Compresses branches in prediction trees.
    
    Reduces memory usage by compressing chains of single-child
    nodes into single edges with multiple items.
    """
    
    def __init__(self, threshold: int = 2):
        """Initialize branch compression.
        
        Args:
            threshold: Minimum chain length to compress.
        """
        self.threshold = threshold
    
    def compress_tree(self, tree: PredictionTree) -> PredictionTree:
        """Compress a prediction tree.
        
        Args:
            tree: Tree to compress.
            
        Returns:
            Compressed tree.
        """
        # Find compressible chains
        chains = self._find_chains(tree.root)
        
        # Compress chains
        for chain in chains:
            if len(chain) >= self.threshold:
                self._compress_chain(chain)
        
        return tree
    
    def _find_chains(self, root: Node) -> List[List[Tuple[Node, int]]]:
        """Find chains of single-child nodes.
        
        Args:
            root: Root node to start from.
            
        Returns:
            List of chains, each chain is a list of (node, item) tuples.
        """
        chains = []
        
        def _find_chain(node: Node, current_chain: List[Tuple[Node, int]]):
            if len(node.children) != 1:
                if len(current_chain) >= self.threshold:
                    chains.append(current_chain)
                return
            
            item, child = next(iter(node.children.items()))
            current_chain.append((node, item))
            _find_chain(child, current_chain)
        
        for item, child in root.children.items():
            _find_chain(child, [(root, item)])
        
        return chains
    
    def _compress_chain(self, chain: List[Tuple[Node, int]]) -> None:
        """Compress a chain of nodes.
        
        Args:
            chain: Chain of (node, item) tuples.
        """
        # Store the sequence of items in the chain
        items = [item for _, item in chain]
        # Mark nodes for removal (simplified implementation)
        # In a full implementation, this would update parent references
    
    def decompress_tree(self, tree: PredictionTree) -> PredictionTree:
        """Decompress a tree (placeholder).
        
        Args:
            tree: Tree to decompress.
            
        Returns:
            Decompressed tree.
        """
        return tree
    
    def __repr__(self) -> str:
        return f"BranchCompression(threshold={self.threshold})"