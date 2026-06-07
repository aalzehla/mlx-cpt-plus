"""Virtual token for compression."""

from dataclasses import dataclass


@dataclass
class VirtualToken:
    """Represents a compressed subsequence as a virtual token.
    
    Virtual tokens allow frequently occurring subsequences to be
    represented as single tokens, reducing memory usage.
    """
    
    id: int
    length: int
    frequency: int = 1
    
    def __post_init__(self):
        """Validate virtual token."""
        if self.length <= 0:
            raise ValueError("Length must be positive")
        if self.frequency < 0:
            raise ValueError("Frequency cannot be negative")
    
    def __repr__(self) -> str:
        return f"VirtualToken(id={self.id}, len={self.length}, freq={self.frequency})"