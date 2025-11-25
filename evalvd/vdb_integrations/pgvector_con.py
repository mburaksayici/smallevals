"""Pgvector Connection to export Chonkie's Chunks into a PostgreSQL database with pgvector using vecs."""

import importlib.util as importutil
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import NAMESPACE_OID, uuid5

from .base import BaseVDBConnection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import vecs
    from sentence_transformers import SentenceTransformer

# Module-level variable that will be set by _import_dependencies
vecs = None


class PgvectorConnection(BaseVDBConnection):
    """Pgvector Connection to export Chonkie's Chunks into a PostgreSQL database with pgvector using vecs.
    
    This handshake allows storing Chonkie chunks in PostgreSQL with vector embeddings
    using the pgvector extension through the vecs client library from Supabase.

    Args:
        client: An existing vecs.Client instance. If provided, other connection parameters are ignored.
        host: PostgreSQL host. Defaults to "localhost".
        port: PostgreSQL port. Defaults to 5432.
        database: PostgreSQL database name. Defaults to "postgres".
        user: PostgreSQL username. Defaults to "postgres".
        password: PostgreSQL password. Defaults to "postgres".
        connection_string: Full PostgreSQL connection string. If provided, individual parameters are ignored.
        collection_name: The name of the collection to store chunks in.
        embedding_model: The embedding model to use for generating embeddings.
        vector_dimensions: The number of dimensions for the vector embeddings.

    """

    def __init__(
        self,
        client: Optional["vecs.Client"] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres", 
        password: str = "postgres",
        connection_string: Optional[str] = None,
        collection_name: str = "chonkie_chunks",
        embedding_model: Optional["SentenceTransformer"] = None,
        vector_dimensions: Optional[int] = None,
    ) -> None:
        """Initialize the Pgvector Connection.
        
        Args:
            client: An existing vecs.Client instance. If provided, other connection parameters are ignored.
            host: PostgreSQL host. Defaults to "localhost".
            port: PostgreSQL port. Defaults to 5432.
            database: PostgreSQL database name. Defaults to "postgres".
            user: PostgreSQL username. Defaults to "postgres".
            password: PostgreSQL password. Defaults to "postgres".
            connection_string: Full PostgreSQL connection string. If provided, individual parameters are ignored.
            collection_name: The name of the collection to store chunks in.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            vector_dimensions: Optional[int]: The number of dimensions for the vector embeddings. Required if embedding_model is not provided.

        """         
        super().__init__()
        
        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize vecs client based on provided parameters
        if client is not None:
            # Use provided client directly
            self.client = client
        elif connection_string is not None:
            # Use provided connection string
            self.client = vecs.create_client(connection_string)
        else:
            # Build connection string from individual parameters
            conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.client = vecs.create_client(conn_str)
        
        self.collection_name = collection_name
        
        # Store embedding model and determine vector dimensions
        self.embedding_model = embedding_model
        if vector_dimensions is not None:
            self.vector_dimensions = vector_dimensions
        elif embedding_model is not None:
            self.vector_dimensions = embedding_model.get_sentence_embedding_dimension()
        else:
            raise ValueError("Either embedding_model or vector_dimensions must be provided")

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            dimension=self.vector_dimensions
        )

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("vecs") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if not self._is_available():
            raise ImportError(
                "vecs is not installed. "
                "Please install it with `pip install chonkie[pgvector]`."
            )
        
        global vecs
        import vecs



    def search(
        self, 
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_value: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: The query text to search for.
            limit: Maximum number of results to return.
            filters: Optional metadata filters in vecs format (e.g., {"year": {"$eq": 2012}}).
            include_metadata: Whether to include metadata in results.
            include_value: Whether to include similarity scores in results.
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with metadata and scores.

        """
        logger.debug(f"Searching PostgreSQL collection: {self.collection_name} with limit={limit}")
        
        # Determine the query embedding
        if query_embedding is None:
            if query is None:
                raise ValueError("Either query or query_embedding must be provided")
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            query_embedding = self.embedding_model.encode(query)

        # Search using vecs
        results = self.collection.query(
            data=query_embedding,
            limit=limit,
            filters=filters,
            include_metadata=include_metadata,
            include_value=include_value
        )

        # Convert vecs results to our format
        formatted_results = []
        for result in results:
            # vecs returns tuples: (id, distance) or (id, distance, metadata)
            result_dict = {"id": result[0]}

            if include_value:
                result_dict["similarity"] = result[1]

            if include_metadata and len(result) > 2:
                metadata = result[2]
                result_dict.update(metadata)

            formatted_results.append(result_dict)

        logger.info(f"Search complete: found {len(formatted_results)} matching chunks")
        return formatted_results

    def create_index(self, method: str = "hnsw", **index_params: Any) -> None:
        """Create a vector index for improved search performance.
        
        Args:
            method: Index method to use. Currently vecs supports various methods.
            **index_params: Additional parameters for the index.

        """
        # Create index using vecs (vecs handles the specifics)
        self.collection.create_index(method=method, **index_params)

        logger.info(f"Created {method} index on collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        # vecs collections have various properties we can inspect
        return {
            "name": self.collection.name,
            "dimension": self.collection.dimension,
            # Add more collection info as available from vecs
        }

    def __repr__(self) -> str:
        """Return the string representation of the PgvectorConnection."""
        return f"PgvectorConnection(collection_name={self.collection_name}, vector_dimensions={self.vector_dimensions})"