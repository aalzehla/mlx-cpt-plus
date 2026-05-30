"""Inverted index for candidate retrieval."""

from typing import Dict, List, Set, Tuple
from collections import defaultdict


class InvertedIndex:
    """Inverted index for efficient candidate retrieval.
    
    Maps items to the sequences they appear in, enabling fast
    retrieval of candidate items for prediction.
    """
    
    def __init__(self):
        """Initialize inverted index."""
        self._index: Dict[int, Set[int]] = defaultdict(set)
        self._item_counts: Dict[int, int] = defaultdict(int)
    
    def add(self, item: int, sequence_id: int) -> None:
        """Add an item to a sequence in the index.
        
        Args:
            item: Item to add.
            sequence_id: ID of the sequence containing the item.
        """
        self._index[item].add(sequence_id)
        self._item_counts[item] += 1
    
    def add_batch(self, items: List[int], sequence_id: int) -> None:
        """Add multiple items to a sequence.
        
        Args:
            items: Items to add.
            sequence_id: ID of the sequence containing the items.
        """
        for item in items:
            self.add(item, sequence_id)
    
    def get_sequences(self, item: int) -> Set[int]:
        """Get sequences containing an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Set of sequence IDs containing the item.
        """
        return self._index.get(item, set())
    
    def get_items(self, sequence_id: int) -> List[int]:
        """Get items in a sequence.
        
        Args:
            sequence_id: ID of the sequence.
            
        Returns:
            List of items in the sequence.
        """
        items = []
        for item, sequences in self._index.items():
            if sequence_id in sequences:
                items.append(item)
        return items
    
    def get_candidate_items(self, context_items: List[int], k: int = 100) -> List[int]:
        """Get candidate items based on context.
        
        Args:
            context_items: Items in the context.
            k: Maximum number of candidates to return.
            
        Returns:
            List of candidate items sorted by frequency.
        """
        candidate_sequences: Set[int] = set()
        for item in context_items:
            candidate_sequences.update(self.get_sequences(item))
        
        candidates = set()
        for seq_id in candidate_sequences:
            candidates.update(self.get_items(seq_id))
        
        # Remove context items from candidates
        candidates -= set(context_items)
        
        # Sort by frequency
        sorted_candidates = sorted(
            candidates,
            key=lambda x: self._item_counts.get(x, 0),
            reverse=True
        )
        return sorted_candidates[:k]
    
    def get_frequency(self, item: int) -> int:
        """Get frequency of an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Frequency count.
        """
        return self._item_counts.get(item, 0)
    
    def size(self) -> int:
        """Get number of unique items.
        
        Returns:
            Number of indexed items.
        """
        return len(self._index)
    
    def clear(self) -> None:
        """Clear the index."""
        self._index.clear()
        self._item_counts.clear()
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"InvertedIndex(size={self.size()})"