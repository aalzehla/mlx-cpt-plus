"""Sequence arena for managing sequences in memory."""

from typing import List, Optional
from mlx_cpt_plus.exceptions import SequenceError


class SequenceArena:
    """Manages sequences in a contiguous memory arena.
    
    This class provides efficient storage and retrieval of sequences
    by storing them in a single contiguous array with offset pointers.
    """
    
    def __init__(self, max_length: int = 1000):
        """Initialize sequence arena.
        
        Args:
            max_length: Maximum length of sequences to store.
        """
        self.max_length = max_length
        self._data: List[int] = []
        self._offsets: List[int] = []
        self._lengths: List[int] = []
    
    def add_sequence(self, sequence: List[int]) -> int:
        """Add a sequence to the arena.
        
        Args:
            sequence: Sequence of integers to add.
            
        Returns:
            Index of the sequence.
            
        Raises:
            SequenceError: If sequence is too long.
        """
        if len(sequence) > self.max_length:
            raise SequenceError(
                f"Sequence length {len(sequence)} exceeds max {self.max_length}"
            )
        
        offset = len(self._data)
        self._data.extend(sequence)
        self._offsets.append(offset)
        self._lengths.append(len(sequence))
        return len(self._offsets) - 1
    
    def get_sequence(self, idx: int) -> Optional[List[int]]:
        """Get a sequence by index.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Sequence if found, None otherwise.
        """
        if idx < 0 or idx >= len(self._offsets):
            return None
        
        offset = self._offsets[idx]
        length = self._lengths[idx]
        return self._data[offset:offset + length]
    
    def get_all_sequences(self) -> List[List[int]]:
        """Get all sequences.
        
        Returns:
            List of all sequences.
        """
        sequences = []
        for i in range(len(self._offsets)):
            sequences.append(self.get_sequence(i))
        return sequences
    
    def clear(self) -> None:
        """Clear all sequences."""
        self._data.clear()
        self._offsets.clear()
        self._lengths.clear()
    
    def size(self) -> int:
        """Get number of sequences.
        
        Returns:
            Number of stored sequences.
        """
        return len(self._offsets)
    
    def memory_usage(self) -> int:
        """Get memory usage in bytes.
        
        Returns:
            Memory usage in bytes.
        """
        return len(self._data) * 8  # Assuming 64-bit integers
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"SequenceArena(size={self.size()}, memory={self.memory_usage()} bytes)"