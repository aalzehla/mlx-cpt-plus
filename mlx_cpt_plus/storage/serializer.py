"""Serializer for model persistence."""

from typing import Any, Dict
import msgpack
import numpy as np


class Serializer:
    """Serializes model data for storage.
    
    Supports multiple serialization formats including msgpack
    and numpy arrays.
    """
    
    def __init__(self, format: str = "msgpack"):
        """Initialize serializer.
        
        Args:
            format: Serialization format.
        """
        self.format = format
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        """Serialize data to bytes.
        
        Args:
            data: Data dictionary to serialize.
            
        Returns:
            Serialized bytes.
        """
        if self.format == "msgpack":
            return msgpack.packb(data, use_bin_type=True)
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def serialize_array(self, array: np.ndarray) -> bytes:
        """Serialize a numpy array.
        
        Args:
            array: Array to serialize.
            
        Returns:
            Serialized bytes.
        """
        return array.tobytes()
    
    def __repr__(self) -> str:
        return f"Serializer(format={self.format!r})"