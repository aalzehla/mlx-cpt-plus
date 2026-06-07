"""Main compressor for CPT+."""

from typing import List, Optional
from mlx_cpt_plus.compression.fsc import FSC
from mlx_cpt_plus.compression.subsequence_dictionary import SubsequenceDictionary
from mlx_cpt_plus.compression.virtual_token import VirtualToken


class Compressor:
    """Main compression engine for CPT+.
    
    Coordinates FSC and subsequence dictionary to compress
    sequences for memory efficiency.
    """
    
    def __init__(
        self,
        min_length: int = 3,
        min_frequency: int = 5,
        max_dictionary_size: int = 10000
    ):
        """Initialize compressor.
        
        Args:
            min_length: Minimum subsequence length.
            min_frequency: Minimum frequency for compression.
            max_dictionary_size: Maximum dictionary size.
        """
        self.fsc = FSC(min_length, min_frequency)
        self.dictionary = SubsequenceDictionary()
        self.max_dictionary_size = max_dictionary_size
    
    def compress(self, sequences: List[List[int]]) -> List[List]:
        """Compress sequences.
        
        Args:
            sequences: Sequences to compress.
            
        Returns:
            Compressed sequences.
        """
        compressed = self.fsc.compress(sequences)
        
        # Add new subsequences to dictionary
        for subseq, id in self.fsc._subsequence_ids.items():
            if self.dictionary.size() < self.max_dictionary_size:
                self.dictionary.add(subseq)
        
        return compressed
    
    def decompress(self, compressed: List[List]) -> List[List[int]]:
        """Decompress sequences.
        
        Args:
            compressed: Compressed sequences.
            
        Returns:
            Decompressed sequences.
        """
        result = []
        for seq in compressed:
            decompressed = []
            for item in seq:
                if isinstance(item, tuple) and item[0] == "SUBSEQ":
                    subseq = self.dictionary.get(item[1])
                    decompressed.extend(subseq)
                else:
                    decompressed.append(item)
            result.append(decompressed)
        return result
    
    def __repr__(self) -> str:
        return f"Compressor(dict_size={self.dictionary.size()})"