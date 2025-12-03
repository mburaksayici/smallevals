"""Base class for Connections."""

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)

# TODO: Move this to inside the BaseVDBConnection class
# Why is this even outside the class?
# def _generate_default_id(*args: Any) -> str:
#     """Generate a default UUID."""
#     return str(uuid.uuid4())


class BaseVDBConnection(ABC):
    """Abstract base class for Connections."""

    @abstractmethod
    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Optional query string. If provided, will be encoded using embedding_model.
            embedding: Optional embedding vector. If provided, query is ignored.
            limit: Number of results to return.

        Returns:
            List of dictionaries with chunk information.
        """
        pass

    @abstractmethod
    def sample_chunks(self, num_chunks: int) -> List[Dict[str, Any]]:
        """
        Sample random chunks from the vector database.
        This is a default implementation that may be overridden by subclasses.

        Args:
            num_chunks: Number of chunks to sample

        Returns:
            List of dictionaries with chunk information:
            [{"text": "...", "metadata": {...}}, ...]
        """

        pass
