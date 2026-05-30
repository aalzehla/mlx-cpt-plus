"""Candidate retrieval for prediction."""

from typing import List, Set, Optional
from mlx_cpt_plus.core.inverted_index import InvertedIndex


class CandidateRetrieval:
    """Retrieves candidate items for prediction.
    
    Uses inverted index and other techniques to efficiently
    retrieve relevant candidate items for next-item prediction.
    """
    
    def __init__(self, index: Optional[InvertedIndex] = None):
        """Initialize candidate retrieval.
        
        Args:
            index: Inverted index to use for retrieval.
        """
        self.index = index or InvertedIndex()
    
    def retrieve(self, context: List[int], k: int = 100) -> List[int]:
        """Retrieve candidate items for a context.
        
        Args:
            context: Context items.
            k: Maximum number of candidates.
            
        Returns:
            List of candidate items.
        """
        return self.index.get_candidate_items(context, k)
    
    def retrieve_with_scores(
        self, context: List[int], k: int = 100
    ) -> List[tuple]:
        """Retrieve candidates with scores.
        
        Args:
            context: Context items.
            k: Maximum number of candidates.
            
        Returns:
            List of (item, score) tuples.
        """
        candidates = self.retrieve(context, k)
        return [(item, self.index.get_frequency(item)) for item in candidates]
    
    def update_index(self, sequences: List[List[int]]) -> None:
        """Update the inverted index with new sequences.
        
        Args:
            sequences: Sequences to index.
        """
        for seq_id, seq in enumerate(sequences):
            self.index.add_batch(seq, seq_id)
    
    def __repr__(self) -> str:
        return f"CandidateRetrieval(index_size={len(self.index)})"