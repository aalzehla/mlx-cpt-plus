"""Exceptions for MLX CPT+."""


class CPTPlusError(Exception):
    """Base exception for CPT+ errors."""
    pass


class ConfigurationError(CPTPlusError):
    """Raised when configuration is invalid."""
    pass


class VocabularyError(CPTPlusError):
    """Raised when vocabulary operations fail."""
    pass


class SequenceError(CPTPlusError):
    """Raised when sequence operations fail."""
    pass


class TreeError(CPTPlusError):
    """Raised when tree operations fail."""
    pass


class PredictionError(CPTPlusError):
    """Raised when prediction operations fail."""
    pass


class StorageError(CPTPlusError):
    """Raised when storage operations fail."""
    pass


class OnlineLearningError(CPTPlusError):
    """Raised when online learning operations fail."""
    pass