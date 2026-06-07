"""MsgPack storage backend."""

from typing import Any, Dict, Optional
import msgpack
import numpy as np
from pathlib import Path


class MsgPackStore:
    """Storage backend using MsgPack format.
    
    Provides file-based storage for model data using MsgPack
    serialization with optional numpy array support.
    """
    
    def __init__(self, path: Optional[str] = None):
        """Initialize MsgPack store.
        
        Args:
            path: Optional file path for storage.
        """
        self.path = path
        self._data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set a value.
        
        Args:
            key: Key to set.
            value: Value to store.
        """
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value.
        
        Args:
            key: Key to look up.
            default: Default value if not found.
            
        Returns:
            Stored value or default.
        """
        return self._data.get(key, default)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save data to file.
        
        Args:
            path: Optional path override.
        """
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            msgpack.pack(self._data, f, use_bin_type=True)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load data from file.
        
        Args:
            path: Optional path override.
        """
        load_path = path or self.path
        if load_path is None:
            raise ValueError("No path specified")
        
        with open(load_path, "rb") as f:
            self._data = msgpack.load(f, raw=False)
    
    def __repr__(self) -> str:
        return f"MsgPackStore(path={self.path!r}, items={len(self._data)})"