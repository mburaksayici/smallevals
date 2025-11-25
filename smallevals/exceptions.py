"""Custom exception classes for EvalVD."""


class EvalVDError(Exception):
    """Base exception for all EvalVD errors."""
    pass


class ModelLoadError(EvalVDError):
    """Raised when model loading fails."""
    pass


class VDBConnectionError(EvalVDError):
    """Raised when vector database connection fails."""
    pass


class MetricsCalculationError(EvalVDError):
    """Raised when metrics calculation fails."""
    pass


class ValidationError(EvalVDError):
    """Raised when input validation fails."""
    pass


class QAGenerationError(EvalVDError):
    """Raised when QA generation fails."""
    pass

