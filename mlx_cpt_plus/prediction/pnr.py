"""Pattern-based Next Recommendation (PNR) for CPT+."""

from typing import List, Dict, Tuple
from collections import defaultdict


class PNR:
    """Pattern-based Next Recommendation.
    
    Extends CPT+ with pattern-based recommendation capabilities,
    identifying common patterns in user behavior for better predictions.
    """
    
    def __init__(self, pattern_length: int = 3, min_support: int = 2):
        """Initialize PNR.
        
        Args:
            pattern_length: Length of patterns to identify.
            min_support: Minimum support for patterns.
        """
        self.pattern_length = pattern_length
        self.min_support = min_support
        self._patterns: Dict[Tuple[int, ...], int] = defaultdict(int)
    
    def learn_patterns(self, sequences: List[List[int]]) -> None:
        """Learn patterns from sequences.
        
        Args:
            sequences: Training sequences.
        """
        self._patterns.clear()
        for seq in sequences:
            self._extract_patterns(seq)
    
    def _extract_patterns(self, seq: List[int]) -> None:
        """Extract patterns from a sequence.
        
        Args:
            seq: Sequence to extract patterns from.
        """
        for i in range(len(seq) - self.pattern_length + 1):
            pattern = tuple(seq[i:i + self.pattern_length])
            self._patterns[pattern] += 1
    
    def get_next_items(self, context: List[int]) -> List[Tuple[int, float]]:
        """Get next item recommendations based on patterns.
        
        Args:
            context: Context items.
            
        Returns:
            List of (item, score) tuples.
        """
        if len(context) < self.pattern_length - 1:
            return []
        
        # Find matching patterns
        matches = []
        for pattern, count in self._patterns.items():
            if pattern[:-1] == tuple(context[-(self.pattern_length - 1):]):
                matches.append((pattern[-1], count))
        
        # Sort by count
        matches.sort(key=lambda x: x[1], reverse=True)
        total = sum(c for _, c in matches) or 1
        
        return [(item, count / total) for item, count in matches]
    
    def get_patterns(self) -> Dict[Tuple[int, ...], int]:
        """Get all learned patterns.
        
        Returns:
            Dictionary of patterns and their counts.
        """
        return dict(self._patterns)
    
    def __repr__(self) -> str:
        return f"PNR(patterns={len(self._patterns)})"