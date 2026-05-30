"""Support table for managing item frequencies."""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SupportTable:
    """Manages support counts for items.
    
    The support table tracks how frequently each item appears
    in the training data, used for filtering and ranking.
    """
    
    min_support: int = 1
    
    _supports: Dict[int, int] = None
    
    def __post_init__(self):
        if self._supports is None:
            self._supports = {}
    
    def increment(self, item: int, count: int = 1) -> None:
        """Increment support count for an item.
        
        Args:
            item: Item to update.
            count: Amount to increment by.
        """
        self._supports[item] = self._supports.get(item, 0) + count
    
    def get_support(self, item: int) -> int:
        """Get support count for an item.
        
        Args:
            item: Item to look up.
            
        Returns:
            Support count.
        """
        return self._supports.get(item, 0)
    
    def set_support(self, item: int, support: int) -> None:
        """Set support count for an item.
        
        Args:
            item: Item to update.
            support: New support count.
        """
        self._supports[item] = support
    
    def get_items_above_threshold(self) -> List[int]:
        """Get items with support above threshold.
        
        Returns:
            List of items with support >= min_support.
        """
        return [
            item for item, support in self._supports.items()
            if support >= self.min_support
        ]
    
    def size(self) -> int:
        """Get number of tracked items.
        
        Returns:
            Number of items in the table.
        """
        return len(self._supports)
    
    def clear(self) -> None:
        """Clear all support counts."""
        self._supports.clear()
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, item: int) -> bool:
        return item in self._supports
    
    def __repr__(self) -> str:
        return f"SupportTable(size={self.size()}, threshold={self.min_support})"