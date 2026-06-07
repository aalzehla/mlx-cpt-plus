"""Frequent Subsequence Compression."""

from typing import List, Dict, Set
from collections import defaultdict


class FSC:
    """Frequent Subsequence Compression.
    
    Identifies and compresses frequent subsequences to reduce
    memory usage in the prediction model.
    """
    
    def __init__(self, min_length: int = 3, min_frequency: int = 5):
        """Initialize FSC.
        
        Args:
            min_length: Minimum subsequence length.
            min_frequency: Minimum frequency for compression.
        """
        self.min_length = min_length
        self.min_frequency = min_frequency
        self._subsequence_counts: Dict[tuple, int] = defaultdict(int)
        self._subsequence_ids: Dict[tuple, int] = {}
        self._next_id = 0
    
    def find_frequent_subsequences(self, sequences: List[List[int]]) -> List[tuple]:
        """Find frequent subsequences in sequences.
        
        Args:
            sequences: List of sequences.
            
        Returns:
            List of frequent subsequences.
        """
        self._subsequence_counts.clear()
        
        for seq in sequences:
            self._scan_sequence(seq)
        
        return [
            subseq for subseq, count in self._subsequence_counts.items()
            if count >= self.min_frequency
        ]
    
    def _scan_sequence(self, seq: List[int]) -> None:
        """Scan a sequence for subsequences.
        
        Args:
            seq: Sequence to scan.
        """
        for length in range(self.min_length, len(seq) + 1):
            for i in range(len(seq) - length + 1):
                subseq = tuple(seq[i:i + length])
                self._subsequence_counts[subseq] += 1
    
    def compress(self, sequences: List[List[int]]) -> List[List]:
        """Compress sequences using frequent subsequences.
        
        Args:
            sequences: Sequences to compress.
            
        Returns:
            Compressed sequences.
        """
        frequent = self.find_frequent_subsequences(sequences)
        
        # Assign IDs to frequent subsequences
        for subseq in frequent:
            if subseq not in self._subsequence_ids:
                self._subsequence_ids[subseq] = self._next_id
                self._next_id += 1
        
        # Replace subsequences with virtual tokens
        compressed = []
        for seq in sequences:
            compressed.append(self._compress_sequence(seq, frequent))
        
        return compressed
    
    def _compress_sequence(self, seq: List[int], frequent: List[tuple]) -> List:
        """Compress a single sequence.
        
        Args:
            seq: Sequence to compress.
            frequent: List of frequent subsequences.
            
        Returns:
            Compressed sequence.
        """
        result = []
        i = 0
        while i < len(seq):
            matched = False
            for subseq in frequent:
                if i + len(subseq) <= len(seq) and seq[i:i + len(subseq)] == list(subseq):
                    result.append(("SUBSEQ", self._subsequence_ids[subseq]))
                    i += len(subseq)
                    matched = True
                    break
            if not matched:
                result.append(seq[i])
                i += 1
        return result
    
    def __repr__(self) -> str:
        return f"FSC(min_len={self.min_length}, min_freq={self.min_frequency})"