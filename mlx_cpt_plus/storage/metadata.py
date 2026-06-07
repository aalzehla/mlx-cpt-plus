"""Metadata for stored models."""

from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime


@dataclass
class Metadata:
    """Metadata for stored models.
    
    Tracks version, creation time, and other metadata
    for model persistence.
    """
    
    version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_type: str = "CPTPlus"
    config: Dict[str, Any] = field(default_factory=dict)
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_type": self.model_type,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metadata":
        """Create from dictionary.
        
        Args:
            data: Dictionary data.
            
        Returns:
            Metadata instance.
        """
        return cls(
            version=data.get("version", "0.1.0"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            model_type=data.get("model_type", "CPTPlus"),
            config=data.get("config", {}),
        )
    
    def __repr__(self) -> str:
        return f"Metadata(version={self.version}, type={self.model_type})"