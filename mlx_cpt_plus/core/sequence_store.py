"""Sequence store for persistent sequence storage."""

from typing import Iterator, List, Optional
from mlx_cpt_plus.exceptions import StorageError


class SequenceStore:
    """Persistent storage for sequences.
    
    This class provides a simple in-memory sequence store that can be
    extended to support persistent storage backends.
    """
    
    def __init__(self):
        """Initialize sequence store."""
        self._sequences: List[List[int]] = []
        self._metadata: dict = {}
    
    def add(self, sequence: List[int]) -> int:
        """Add a sequence to the store.
        
        Args:
            sequence: Sequence to add.
            
        Returns:
            Index of the added sequence.
        """
        self._sequences.append(sequence.copy())
        return len(self._sequences) - 1
    
    def get(self, idx: int) -> Optional[List[int]]:
        """Get a sequence by index.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Sequence if found, None otherwise.
        """
        if 0 <= idx < len(self._sequences):
            return self._sequences[idx]
        return None
    
    def update(self, idx: int, sequence: List[int]) -> bool:
        """Update a sequence.
        
        Args:
            idx: Index of the sequence.
            sequence: New sequence value.
            
        Returns:
            True if updated, False if index not found.
        """
        if 0 <= idx < len(self._sequences):
            self._sequences[idx] = sequence.copy()
            return True
        return False
    
    def remove(self, idx: int) -> bool:
        """Remove a sequence.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            True if removed, False if index not found.
        """
        if 0 <= idx < len(self._sequences):
            self._sequences.pop(idx)
            return True
        return False
    
    def count(self) -> int:
        """Get number of sequences.
        
        Returns:
            Number of stored sequences.
        """
        return len(self._sequences)
    
    def clear(self) -> None:
        """Clear all sequences."""
        self._sequences.clear()
    
    def iter_sequences(self) -> Iterator[List[int]]:
        """Iterate over all sequences.
        
        Yields:
            Each sequence in the store.
        """
        for seq in self._sequences:
            yield seq
    
    def __len__(self) -> int:
        return self.count()
    
    def __repr__(self) -> str:
        return f"SequenceStore(count={self.count()})"