"""Top-k prediction results."""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TopK:
    """Holds top-k prediction results.
    
    Provides efficient storage and access to top-k predictions.
    """
    
    items: List[int]
    scores: List[float]
    k: int = 10
    
    def __post_init__(self):
        if len(self.items) != len(self.scores):
            raise ValueError("items and scores must have same length")
        self.k = min(self.k, len(self.items))
    
    def get_top_k(self, k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Get top-k items with scores.
        
        Args:
            k: Number of items to return (default: self.k).
            
        Returns:
            List of (item, score) tuples.
        """
        k = k or self.k
        return list(zip(self.items[:k], self.scores[:k]))
    
    def get_item_score(self, item: int) -> Optional[float]:
        """Get score for an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Score if found, None otherwise.
        """
        try:
            idx = self.items.index(item)
            return self.scores[idx]
        except ValueError:
            return None
    
    def contains(self, item: int) -> bool:
        """Check if item is in top-k.
        
        Args:
            item: Item to check.
            
        Returns:
            True if item is in top-k.
        """
        return item in self.items[:self.k]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return f"TopK(k={self.k}, items={len(self.items)})"