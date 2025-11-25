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
        limit: int = 5,
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

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query vector database with a question and return relevant chunks.
        This is a convenience method that calls search().

        Args:
            question: Query question
            top_k: Number of chunks to retrieve

        Returns:
            List of dictionaries with chunk information:
            [{"text": "...", "metadata": {...}, "score": ...}, ...]
        """
        results = self.search(query=question, limit=top_k)
        # Normalize results to expected format
        normalized = []
        for result in results:
            if isinstance(result, dict):
                # Ensure required fields exist
                normalized.append({
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", result.get("similarity", None)),
                    "id": result.get("id", None),
                })
            else:
                normalized.append({"text": str(result), "metadata": {}, "score": None})
        return normalized

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
        # Default implementation: query with random queries
        import random
        random_queries = [
            "information", "data", "content", "document", "text",
            "details", "facts", "knowledge", "content", "text"
        ]
        results = []
        for query in random_queries:
            chunk_results = self.query(query, top_k=max(1, num_chunks // len(random_queries)))
            results.extend(chunk_results)
            if len(results) >= num_chunks:
                break
        return results[:num_chunks]

    def __call__(self, chunks: Union[Any, List[Any]]) -> Any:
        """Write chunks using the default batch method when the instance is called.

        Args:
            chunks (Union[Any, List[Any]]): A single chunk or a sequence of chunks.

        Returns:
            Any: The result from the database write operation.

        """
        if isinstance(chunks, (dict, str)) or isinstance(chunks, Sequence):
            chunk_count = 1 if not isinstance(chunks, Sequence) else len(chunks)
            logger.info(f"Writing {chunk_count} chunk(s) to database with {self.__class__.__name__}")
            try:
                result = self.write(chunks)
                logger.debug(f"Successfully wrote {chunk_count} chunk(s)")
                return result
            except Exception as e:
                logger.error(
                    f"Failed to write {chunk_count} chunk(s) to database",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        else:
            raise TypeError("Input must be a Chunk or a sequence of Chunks.")