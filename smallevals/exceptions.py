"""Custom exception classes for smallevals."""


class smallevalsError(Exception):
    """Base exception for all smallevals errors."""
    pass


class ModelLoadError(smallevalsError):
    """Raised when model loading fails."""
    pass


class VDBConnectionError(smallevalsError):
    """Raised when vector database connection fails."""
    pass


class MetricsCalculationError(smallevalsError):
    """Raised when metrics calculation fails."""
    pass


class ValidationError(smallevalsError):
    """Raised when input validation fails."""
    pass


class QAGenerationError(smallevalsError):
    """Raised when QA generation fails."""
    pass

