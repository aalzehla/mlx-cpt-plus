"""Similarity computation for prediction."""

from typing import List, Tuple
import numpy as np


class Similarity:
    """Computes similarity between sequences and contexts.
    
    Provides various similarity metrics for comparing
    sequences in prediction tasks.
    """
    
    def __init__(self, method: str = "jaccard"):
        """Initialize similarity calculator.
        
        Args:
            method: Similarity method ('jaccard', 'cosine', 'overlap').
        """
        self.method = method
    
    def jaccard(self, set1: List[int], set2: List[int]) -> float:
        """Compute Jaccard similarity.
        
        Args:
            set1: First set of items.
            set2: Second set of items.
            
        Returns:
            Jaccard similarity score.
        """
        s1, s2 = set(set1), set(set2)
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0
    
    def cosine(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Cosine similarity score.
        """
        v1, v2 = np.array(vec1), np.array(vec2)
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0
    
    def overlap(self, list1: List[int], list2: List[int]) -> float:
        """Compute overlap coefficient.
        
        Args:
            list1: First list.
            list2: Second list.
            
        Returns:
            Overlap coefficient.
        """
        s1, s2 = set(list1), set(list2)
        min_size = min(len(s1), len(s2))
        return len(s1 & s2) / min_size if min_size > 0 else 0.0
    
    def compute(self, a, b, **kwargs) -> float:
        """Compute similarity using configured method.
        
        Args:
            a: First input.
            b: Second input.
            **kwargs: Additional arguments.
            
        Returns:
            Similarity score.
        """
        if self.method == "jaccard":
            return self.jaccard(a, b)
        elif self.method == "cosine":
            return self.cosine(a, b)
        elif self.method == "overlap":
            return self.overlap(a, b)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
    
    def __repr__(self) -> str:
        return f"Similarity(method={self.method!r})"