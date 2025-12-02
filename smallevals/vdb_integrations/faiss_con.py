"""FAISS Connection to export Chonkie's Chunks into a FAISS index."""

import importlib.util
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from .base import BaseVDBConnection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


class FaissConnection(BaseVDBConnection):
    """FAISS Connection to manage chunks in a FAISS index.

    This creates an in-memory FAISS index with a defined schema,
    and enables similarity search over chunks.

    Usage:

        from sentence_transformers import SentenceTransformer
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        conn = FaissConnection(
            embedding_model=embedding_model,
            dimension=384,
        )

    Args:
        embedding_model: The embedding model to use for encoding queries.
        dimension: The dimension of the embeddings.
        index_type: Type of FAISS index to use. Default is "Flat" (IndexFlatL2).
        metric: Distance metric to use. Default is "L2".
    """

    def __init__(
        self,
        embedding_model: Optional["SentenceTransformer"] = None,
        dimension: int = 384,
        index_type: str = "Flat",
        metric: str = "L2",
        **kwargs: Any,
    ) -> None:
        self._import_dependencies()
        super().__init__()

        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        # Create FAISS index
        self.index = self._create_index()
        
        # Store chunk data separately (FAISS only stores vectors)
        # Maps FAISS internal ID (int) -> chunk data dict
        self._chunk_data: List[Dict[str, Any]] = []
        
        logger.info(f"Created FAISS index: {index_type} with {metric} metric, dimension={dimension}")

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global faiss
            import faiss
        else:
            raise ImportError(
                "FAISS is not installed. "
                "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )

    def _is_available(self) -> bool:
        """Check if faiss is installed."""
        return importlib.util.find_spec("faiss") is not None

    def _create_index(self) -> "faiss.Index":
        """Create a FAISS index based on the specified type and metric."""
        if self.index_type == "Flat":
            if self.metric == "L2":
                return faiss.IndexFlatL2(self.dimension)
            elif self.metric == "IP":  # Inner Product
                return faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def __repr__(self) -> str:
        return f"FaissConnection(dimension={self.dimension}, index_type={self.index_type}, n_vectors={self.index.ntotal})"

    def add(
        self,
        texts: List[str],
        embeddings: Union[List[List[float]], "np.ndarray"],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add chunks to the FAISS index.

        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors or numpy array.
            metadatas: Optional list of metadata dictionaries for each chunk.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")

        # Convert embeddings to numpy array if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        # Validate dimensions
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
            )

        # Add vectors to index
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # Store chunk data
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunk_data = {
                "id": start_id + i,
                "text": text,
                "start_index": metadata.get("start_index", 0),
                "end_index": metadata.get("end_index", len(text)),
                "token_count": metadata.get("token_count", len(text.split())),
            }
            # Add any additional metadata
            for key, value in metadata.items():
                if key not in chunk_data:
                    chunk_data[key] = value
            self._chunk_data.append(chunk_data)

        logger.info(f"Added {len(texts)} chunks to FAISS index. Total: {self.index.ntotal}")

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], "np.ndarray"]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar chunks to the query.

        Args:
            query: Optional query string. If provided, will be encoded using embedding_model.
            embedding: Optional embedding vector. If provided, query is ignored.
            limit: Number of results to return.

        Returns:
            List of dictionaries containing chunk information and scores.
        """
        if embedding is None and query is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        # Encode query text if needed
        if query:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            query_embedding = self.embedding_model.encode(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
        else:
            # Convert embedding to numpy array
            if isinstance(embedding, np.ndarray):
                query_vector = embedding.astype(np.float32)
            else:
                query_vector = np.array([embedding], dtype=np.float32)
            
            # Ensure 2D shape
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

        # Validate dimensions
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_vector.shape[1]} does not match index dimension {self.dimension}"
            )

        # Check if index has any vectors
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, returning empty results")
            return []

        # Search
        k = min(limit, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        # Build results
        matches: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            # Get chunk data
            chunk = self._chunk_data[idx].copy()
            
            # Add distance/score
            distance = float(distances[0][i])
            chunk["score"] = distance  # L2 distance (lower is better)
            chunk["distance"] = distance
            
            matches.append(chunk)

        return matches

    def sample_chunks(self, num_chunks: int) -> List[Dict[str, Any]]:
        """Sample random chunks from the FAISS index.

        Args:
            num_chunks: Number of chunks to sample

        Returns:
            List of dictionaries with chunk information.
        """
        import random
        
        if self.index.ntotal == 0:
            return []
        
        total_chunks = len(self._chunk_data)
        num_to_sample = min(num_chunks, total_chunks)
        
        # Sample random indices
        if num_to_sample >= total_chunks:
            indices = list(range(total_chunks))
        else:
            indices = random.sample(range(total_chunks), num_to_sample)
        
        # Build results
        chunks = []
        for idx in indices:
            chunk_data = self._chunk_data[idx].copy()
            # Ensure metadata field exists for compatibility
            if "metadata" not in chunk_data:
                chunk_data["metadata"] = {}
            chunks.append(chunk_data)
        
        return chunks

