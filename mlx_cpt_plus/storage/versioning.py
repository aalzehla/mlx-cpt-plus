"""Versioning for model persistence."""

from typing import Tuple


class Versioning:
    """Manages model versioning.
    
    Provides version comparison and migration support
    for stored models.
    """
    
    CURRENT_VERSION = "0.1.0"
    
    @staticmethod
    def parse_version(version: str) -> Tuple[int, ...]:
        """Parse version string to tuple.
        
        Args:
            version: Version string (e.g., "0.1.0").
            
        Returns:
            Version tuple (e.g., (0, 1, 0)).
        """
        return tuple(int(x) for x in version.split("."))
    
    @staticmethod
    def is_compatible(version: str) -> bool:
        """Check if version is compatible.
        
        Args:
            version: Version to check.
            
        Returns:
            True if compatible.
        """
        return Versioning.parse_version(version) <= Versioning.parse_version(Versioning.CURRENT_VERSION)
    
    @staticmethod
    def needs_migration(version: str) -> bool:
        """Check if migration is needed.
        
        Args:
            version: Version to check.
            
        Returns:
            True if migration needed.
        """
        return version != Versioning.CURRENT_VERSION
    
    def migrate(self, data: dict, from_version: str) -> dict:
        """Migrate data to current version.
        
        Args:
            data: Data to migrate.
            from_version: Source version.
            
        Returns:
            Migrated data.
        """
        # Simple migration - in production would have version-specific handlers
        return data
    
    def __repr__(self) -> str:
        return f"Versioning(current={self.CURRENT_VERSION})"