"""Vocabulary for managing tokens."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from mlx_cpt_plus.exceptions import VocabularyError


@dataclass
class Vocabulary:
    """Manages vocabulary for sequence modeling.
    
    The vocabulary maintains a mapping between tokens and their indices,
    along with frequency counts for filtering and pruning.
    """
    
    min_frequency: int = 2
    max_size: int = 1_000_000
    
    _token_to_idx: Dict[str, int] = field(default_factory=dict)
    _idx_to_token: Dict[int, str] = field(default_factory=dict)
    _frequencies: Dict[str, int] = field(default_factory=dict)
    _next_idx: int = 0
    
    def add_token(self, token: str, frequency: int = 1) -> int:
        """Add a token to the vocabulary.
        
        Args:
            token: Token to add.
            frequency: Frequency count for the token.
            
        Returns:
            Index of the token.
            
        Raises:
            VocabularyError: If vocabulary is full.
        """
        if token in self._token_to_idx:
            self._frequencies[token] += frequency
            return self._token_to_idx[token]
        
        if len(self._token_to_idx) >= self.max_size:
            raise VocabularyError(f"Vocabulary size limit ({self.max_size}) reached")
        
        idx = self._next_idx
        self._token_to_idx[token] = idx
        self._idx_to_token[idx] = token
        self._frequencies[token] = frequency
        self._next_idx += 1
        return idx
    
    def get_index(self, token: str) -> Optional[int]:
        """Get index for a token.
        
        Args:
            token: Token to look up.
            
        Returns:
            Index if found, None otherwise.
        """
        return self._token_to_idx.get(token)
    
    def get_token(self, idx: int) -> Optional[str]:
        """Get token for an index.
        
        Args:
            idx: Index to look up.
            
        Returns:
            Token if found, None otherwise.
        """
        return self._idx_to_token.get(idx)
    
    def get_frequency(self, token: str) -> int:
        """Get frequency of a token.
        
        Args:
            token: Token to look up.
            
        Returns:
            Frequency count.
        """
        return self._frequencies.get(token, 0)
    
    def size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary.
        """
        return len(self._token_to_idx)
    
    def prune(self) -> int:
        """Remove tokens below minimum frequency.
        
        Returns:
            Number of tokens removed.
        """
        to_remove = [
            token for token, freq in self._frequencies.items()
            if freq < self.min_frequency
        ]
        
        for token in to_remove:
            idx = self._token_to_idx.pop(token)
            self._idx_to_token.pop(idx)
            self._frequencies.pop(token)
        
        return len(to_remove)
    
    def build_from_sequences(self, sequences: List[List[str]]) -> None:
        """Build vocabulary from sequences of tokens.
        
        Args:
            sequences: List of token sequences.
        """
        for seq in sequences:
            for token in seq:
                self.add_token(token)
        self.prune()
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, token: str) -> bool:
        return token in self._token_to_idx
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size()}, min_freq={self.min_frequency})"