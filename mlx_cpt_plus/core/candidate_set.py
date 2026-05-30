"""Candidate set for prediction."""

from typing import List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CandidateSet:
    """Set of candidates for prediction.
    
    Manages a collection of candidate items with their scores
    for ranking and selection.
    """
    
    items: List[int] = None
    scores: List[float] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
        if self.scores is None:
            self.scores = []
    
    def add(self, item: int, score: float) -> None:
        """Add a candidate.
        
        Args:
            item: Candidate item.
            score: Score for the candidate.
        """
        self.items.append(item)
        self.scores.append(score)
    
    def get_top_k(self, k: int) -> List[Tuple[int, float]]:
        """Get top-k candidates by score.
        
        Args:
            k: Number of top candidates to return.
            
        Returns:
            List of (item, score) tuples.
        """
        import operator
        paired = list(zip(self.items, self.scores))
        paired.sort(key=operator.itemgetter(1), reverse=True)
        return paired[:k]
    
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
    
    def size(self) -> int:
        """Get number of candidates.
        
        Returns:
            Number of candidates.
        """
        return len(self.items)
    
    def clear(self) -> None:
        """Clear all candidates."""
        self.items.clear()
        self.scores.clear()
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"CandidateSet(size={self.size()})"