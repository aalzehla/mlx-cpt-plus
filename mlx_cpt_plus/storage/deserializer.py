"""Deserializer for model persistence."""

from typing import Any, Dict
import msgpack
import numpy as np


class Deserializer:
    """Deserializes model data from storage.
    
    Supports multiple deserialization formats including msgpack
    and numpy arrays.
    """
    
    def __init__(self, format: str = "msgpack"):
        """Initialize deserializer.
        
        Args:
            format: Deserialization format.
        """
        self.format = format
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to data.
        
        Args:
            data: Serialized bytes.
            
        Returns:
            Data dictionary.
        """
        if self.format == "msgpack":
            return msgpack.unpackb(data, raw=False)
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def deserialize_array(self, data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
        """Deserialize a numpy array.
        
        Args:
            data: Serialized array bytes.
            dtype: Array data type.
            shape: Array shape.
            
        Returns:
            Deserialized array.
        """
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    def __repr__(self) -> str:
        return f"Deserializer(format={self.format!r})"