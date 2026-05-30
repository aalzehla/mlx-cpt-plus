"""Ranking for prediction results."""

from typing import List, Tuple, Optional
import numpy as np


class Ranking:
    """Ranks prediction candidates.
    
    Provides various ranking algorithms to order
    candidate items by their prediction scores.
    """
    
    def __init__(self, method: str = "score"):
        """Initialize ranking mechanism.
        
        Args:
            method: Ranking method ('score', 'normalized', 'diversity').
        """
        self.method = method
    
    def rank_by_score(
        self, candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Rank by raw scores.
        
        Args:
            candidates: List of (item, score) tuples.
            
        Returns:
            Sorted candidates by score.
        """
        return sorted(candidates, key=lambda x: x[1], reverse=True)
    
    def rank_normalized(
        self, candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Rank with normalized scores.
        
        Args:
            candidates: List of (item, score) tuples.
            
        Returns:
            Sorted candidates with normalized scores.
        """
        if not candidates:
            return []
        
        scores = np.array([s for _, s in candidates])
        min_s, max_s = scores.min(), scores.max()
        
        if max_s == min_s:
            return [(item, 0.5) for item, _ in candidates]
        
        normalized = (scores - min_s) / (max_s - min_s)
        return sorted(
            zip([item for item, _ in candidates], normalized),
            key=lambda x: x[1],
            reverse=True
        )
    
    def rank_diversity(
        self,
        candidates: List[Tuple[int, float]],
        context: List[int],
        diversity_weight: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Rank with diversity penalty.
        
        Args:
            candidates: List of (item, score) tuples.
            context: Context items.
            diversity_weight: Weight for diversity component.
            
        Returns:
            Sorted candidates with diversity-adjusted scores.
        """
        context_set = set(context)
        adjusted = []
        
        for item, score in candidates:
            # Penalize items in context
            penalty = 1.0 if item in context_set else 0.0
            adjusted_score = score * (1.0 - diversity_weight * penalty)
            adjusted.append((item, adjusted_score))
        
        return sorted(adjusted, key=lambda x: x[1], reverse=True)
    
    def rank(
        self,
        candidates: List[Tuple[int, float]],
        context: Optional[List[int]] = None,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """Rank candidates using configured method.
        
        Args:
            candidates: List of (item, score) tuples.
            context: Optional context for diversity ranking.
            **kwargs: Additional arguments.
            
        Returns:
            Sorted candidates.
        """
        if self.method == "score":
            return self.rank_by_score(candidates)
        elif self.method == "normalized":
            return self.rank_normalized(candidates)
        elif self.method == "diversity":
            return self.rank_diversity(candidates, context or [], **kwargs)
        else:
            raise ValueError(f"Unknown ranking method: {self.method}")
    
    def __repr__(self) -> str:
        return f"Ranking(method={self.method!r})"