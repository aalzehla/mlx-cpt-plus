"""Subsequence dictionary for compression."""

from typing import List, Dict, Tuple


class SubsequenceDictionary:
    """Dictionary of subsequences for compression.
    
    Stores frequently occurring subsequences and their
    compressed representations.
    """
    
    def __init__(self):
        """Initialize subsequence dictionary."""
        self._subsequences: Dict[int, Tuple[int, ...]] = {}
        self._reverse_index: Dict[Tuple[int, ...], int] = {}
        self._next_id = 0
    
    def add(self, subsequence: Tuple[int, ...]) -> int:
        """Add a subsequence to the dictionary.
        
        Args:
            subsequence: Subsequence to add.
            
        Returns:
            ID of the subsequence.
        """
        if subsequence in self._reverse_index:
            return self._reverse_index[subsequence]
        
        id = self._next_id
        self._subsequences[id] = subsequence
        self._reverse_index[subsequence] = id
        self._next_id += 1
        return id
    
    def get(self, id: int) -> Tuple[int, ...]:
        """Get subsequence by ID.
        
        Args:
            id: Subsequence ID.
            
        Returns:
            Subsequence tuple.
        """
        return self._subsequences.get(id, ())
    
    def lookup(self, subsequence: Tuple[int, ...]) -> int:
        """Look up subsequence ID.
        
        Args:
            subsequence: Subsequence to look up.
            
        Returns:
            ID if found, -1 otherwise.
        """
        return self._reverse_index.get(subsequence, -1)
    
    def size(self) -> int:
        """Get dictionary size.
        
        Returns:
            Number of subsequences.
        """
        return len(self._subsequences)
    
    def clear(self) -> None:
        """Clear the dictionary."""
        self._subsequences.clear()
        self._reverse_index.clear()
        self._next_id = 0
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"SubsequenceDictionary(size={self.size()})"