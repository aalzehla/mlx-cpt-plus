"""Voting mechanism for prediction."""

from typing import List, Dict, Tuple
from collections import Counter


class Voting:
    """Implements voting for prediction.
    
    Aggregates votes from multiple sources to determine
    the most likely next items.
    """
    
    def __init__(self, method: str = "weighted"):
        """Initialize voting mechanism.
        
        Args:
            method: Voting method ('simple', 'weighted', 'ranked').
        """
        self.method = method
    
    def simple_vote(self, candidates: List[Tuple[int, float]]) -> Dict[int, int]:
        """Simple voting - count occurrences.
        
        Args:
            candidates: List of (item, score) tuples.
            
        Returns:
            Vote counts per item.
        """
        votes = Counter()
        for item, _ in candidates:
            votes[item] += 1
        return dict(votes)
    
    def weighted_vote(self, candidates: List[Tuple[int, float]]) -> Dict[int, float]:
        """Weighted voting - sum scores.
        
        Args:
            candidates: List of (item, score) tuples.
            
        Returns:
            Weighted scores per item.
        """
        scores = {}
        for item, score in candidates:
            scores[item] = scores.get(item, 0.0) + score
        return scores
    
    def ranked_vote(
        self, candidate_lists: List[List[int]], weights: List[float] = None
    ) -> Dict[int, float]:
        """Ranked voting - Borda count.
        
        Args:
            candidate_lists: List of ranked candidate lists.
            weights: Optional weights for each list.
            
        Returns:
            Borda scores per item.
        """
        if weights is None:
            weights = [1.0] * len(candidate_lists)
        
        scores = {}
        for candidates, weight in zip(candidate_lists, weights):
            for rank, item in enumerate(candidates):
                # Borda count: points = n - rank
                points = len(candidates) - rank
                scores[item] = scores.get(item, 0.0) + points * weight
        return scores
    
    def vote(
        self, candidates: List[Tuple[int, float]], **kwargs
    ) -> List[Tuple[int, float]]:
        """Perform voting using configured method.
        
        Args:
            candidates: List of (item, score) tuples.
            **kwargs: Additional arguments.
            
        Returns:
            Sorted list of (item, score) tuples.
        """
        if self.method == "simple":
            scores = self.simple_vote(candidates)
        elif self.method == "weighted":
            scores = self.weighted_vote(candidates)
        else:
            raise ValueError(f"Unknown voting method: {self.method}")
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def __repr__(self) -> str:
        return f"Voting(method={self.method!r})"