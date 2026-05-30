"""Alphabet for token encoding."""

from typing import Dict, Optional


class Alphabet:
    """Manages the alphabet for token encoding.
    
    The alphabet defines the set of characters/symbols used for encoding
    sequences into numerical representations.
    """
    
    def __init__(self, name: str = "default"):
        """Initialize alphabet.
        
        Args:
            name: Name of the alphabet.
        """
        self.name = name
        self._char_to_idx: Dict[str, int] = {}
        self._idx_to_char: Dict[int, str] = {}
        self._next_idx = 0
    
    def add_char(self, char: str) -> int:
        """Add a character to the alphabet.
        
        Args:
            char: Character to add.
            
        Returns:
            Index of the character.
        """
        if char in self._char_to_idx:
            return self._char_to_idx[char]
        
        idx = self._next_idx
        self._char_to_idx[char] = idx
        self._idx_to_char[idx] = char
        self._next_idx += 1
        return idx
    
    def get_index(self, char: str) -> Optional[int]:
        """Get index for a character.
        
        Args:
            char: Character to look up.
            
        Returns:
            Index if found, None otherwise.
        """
        return self._char_to_idx.get(char)
    
    def get_char(self, idx: int) -> Optional[str]:
        """Get character for an index.
        
        Args:
            idx: Index to look up.
            
        Returns:
            Character if found, None otherwise.
        """
        return self._idx_to_char.get(idx)
    
    def size(self) -> int:
        """Get the size of the alphabet.
        
        Returns:
            Number of characters in the alphabet.
        """
        return len(self._char_to_idx)
    
    def encode(self, text: str) -> list:
        """Encode a string to indices.
        
        Args:
            text: String to encode.
            
        Returns:
            List of indices.
        """
        return [self.add_char(c) for c in text]
    
    def decode(self, indices: list) -> str:
        """Decode indices to a string.
        
        Args:
            indices: List of indices.
            
        Returns:
            Decoded string.
        """
        chars = []
        for idx in indices:
            char = self.get_char(idx)
            if char is not None:
                chars.append(char)
        return "".join(chars)
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"Alphabet(name={self.name!r}, size={self.size()})"