"""Lookup table for efficient indexing."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class LookupTable:
    """Efficient lookup table for key-value mappings.
    
    This class provides a high-performance lookup table that can be used
    for mapping keys to values, with support for batch operations.
    """
    
    def __init__(self, initial_capacity: int = 1000):
        """Initialize lookup table.
        
        Args:
            initial_capacity: Initial capacity for the table.
        """
        self._keys: Dict[int, int] = {}
        self._values: Dict[int, float] = {}
        self._initial_capacity = initial_capacity
    
    def put(self, key: int, value: float) -> None:
        """Put a key-value pair.
        
        Args:
            key: Key to insert.
            value: Value to associate with the key.
        """
        self._keys[key] = key
        self._values[key] = value
    
    def get(self, key: int, default: Optional[float] = None) -> Optional[float]:
        """Get value for a key.
        
        Args:
            key: Key to look up.
            default: Default value if key not found.
            
        Returns:
            Value if found, default otherwise.
        """
        return self._values.get(key, default)
    
    def contains(self, key: int) -> bool:
        """Check if key exists.
        
        Args:
            key: Key to check.
            
        Returns:
            True if key exists, False otherwise.
        """
        return key in self._values
    
    def remove(self, key: int) -> bool:
        """Remove a key-value pair.
        
        Args:
            key: Key to remove.
            
        Returns:
            True if removed, False if key not found.
        """
        if key in self._values:
            del self._keys[key]
            del self._values[key]
            return True
        return False
    
    def size(self) -> int:
        """Get number of entries.
        
        Returns:
            Number of key-value pairs.
        """
        return len(self._values)
    
    def keys(self) -> List[int]:
        """Get all keys.
        
        Returns:
            List of all keys.
        """
        return list(self._values.keys())
    
    def values(self) -> List[float]:
        """Get all values.
        
        Returns:
            List of all values.
        """
        return list(self._values.values())
    
    def items(self) -> List[Tuple[int, float]]:
        """Get all key-value pairs.
        
        Returns:
            List of (key, value) tuples.
        """
        return list(self._values.items())
    
    def clear(self) -> None:
        """Clear all entries."""
        self._keys.clear()
        self._values.clear()
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays.
        
        Returns:
            Tuple of (keys_array, values_array).
        """
        keys = np.array(self.keys(), dtype=np.int64)
        values = np.array(self.values(), dtype=np.float32)
        return keys, values
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, key: int) -> bool:
        return self.contains(key)
    
    def __repr__(self) -> str:
        return f"LookupTable(size={self.size()})"