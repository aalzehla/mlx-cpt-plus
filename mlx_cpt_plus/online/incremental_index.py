"""Incremental index for online learning."""

from typing import List, Set, Dict
from collections import defaultdict


class IncrementalIndex:
    """Incremental index for online learning.
    
    Maintains an index that can be updated incrementally
    as new data arrives.
    """
    
    def __init__(self):
        """Initialize incremental index."""
        self._item_index: Dict[int, Set[int]] = defaultdict(set)
        self._sequence_count = 0
    
    def add_sequence(self, sequence: List[int]) -> None:
        """Add a sequence to the index.
        
        Args:
            sequence: Sequence to add.
        """
        seq_id = self._sequence_count
        self._sequence_count += 1
        
        for item in sequence:
            self._item_index[item].add(seq_id)
    
    def get_sequences_for_item(self, item: int) -> Set[int]:
        """Get sequences containing an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Set of sequence IDs.
        """
        return self._item_index.get(item, set())
    
    def get_items_for_sequence(self, seq_id: int) -> List[int]:
        """Get items in a sequence.
        
        Args:
            seq_id: Sequence ID.
            
        Returns:
            List of items.
        """
        items = []
        for item, seqs in self._item_index.items():
            if seq_id in seqs:
                items.append(item)
        return items
    
    def update(self, sequences: List[List[int]]) -> None:
        """Update index with new sequences.
        
        Args:
            sequences: Sequences to add.
        """
        for seq in sequences:
            self.add_sequence(seq)
    
    def size(self) -> int:
        """Get number of indexed items.
        
        Returns:
            Number of unique items.
        """
        return len(self._item_index)
    
    def clear(self) -> None:
        """Clear the index."""
        self._item_index.clear()
        self._sequence_count = 0
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"IncrementalIndex(items={self.size()}, sequences={self._sequence_count})"