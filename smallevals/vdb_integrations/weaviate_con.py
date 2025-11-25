"""Weaviate Connection to export Chonkie's Chunks into a Weaviate collection."""

import importlib.util as importutil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
import logging
from urllib.parse import urlparse
from uuid import NAMESPACE_OID, uuid5

from .base import BaseVDBConnection
from .utils import generate_random_collection_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import weaviate
    from sentence_transformers import SentenceTransformer


class WeaviateConnection(BaseVDBConnection):
    """Weaviate Connection to export Chonkie's Chunks into a Weaviate collection.

    This handshake allows storing Chonkie chunks in Weaviate with vector embeddings.
    It supports both API key and OAuth authentication methods.

    Args:
        client: Optional[weaviate.Client]: An existing Weaviate client instance.
        collection_name: Union[str, Literal["random"]]: The name of the collection to use.
        embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
        url: Optional[str]: The URL to the Weaviate server.
        api_key: Optional[str]: The API key for authentication.
        auth_config: Optional[Dict[str, Any]]: OAuth configuration for authentication.
        batch_size: int: The batch size for batch operations. Defaults to 100.
        batch_dynamic: bool: Whether to use dynamic batching. Defaults to True.
        batch_timeout_retries: int: Number of retries for batch timeouts. Defaults to 3.
        additional_headers: Optional[Dict[str, str]]: Additional headers for the Weaviate client.

    """

    def __init__(
        self,
        client: Optional[Any] = None,  # weaviate.Client
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        batch_dynamic: bool = True,
        batch_timeout_retries: int = 3,
        additional_headers: Optional[Dict[str, str]] = None,
        http_secure: bool = False,
        grpc_host: Optional[str] = None,
        grpc_port: int = 50051,
        grpc_secure: bool = False,
    ) -> None:
        """Initialize the Weaviate Connection.

        Args:
            client: Optional[weaviate.Client]: An existing Weaviate client instance.
            collection_name: Union[str, Literal["random"]]: The name of the collection to use.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            url: Optional[str]: The URL to the Weaviate server.
            api_key: Optional[str]: The API key for authentication.
            auth_config: Optional[Dict[str, Any]]: OAuth configuration for authentication.
            batch_size: int: The batch size for batch operations. Defaults to 100.
            batch_dynamic: bool: Whether to use dynamic batching. Defaults to True.
            batch_timeout_retries: int: Number of retries for batch timeouts. Defaults to 3.
            additional_headers: Optional[Dict[str, str]]: Additional headers for the Weaviate client.
            http_secure: bool: Whether to use HTTPS for HTTP connections. Defaults to False.
            grpc_host: Optional[str]: The host for gRPC connections. Defaults to the same as HTTP host.
            grpc_port: int: The port for gRPC connections. Defaults to 50051.
            grpc_secure: bool: Whether to use a secure channel for gRPC connections. Defaults to False.

        """
        super().__init__()

        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize the Weaviate client
        if client is None:
            if url is None:
                url = "http://localhost:8080"

            try:
                # connect to cloud client
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.Auth.api_key(
                        api_key if api_key is not None else ""
                    ),
                )
            except Exception:
                # connect to a localhost
                # Parse the URL to get the host and port
                parsed_url = urlparse(url)
                host = parsed_url.hostname or "localhost"
                port = parsed_url.port or 8080

                auth_credentials: Optional[Any] = None
                if api_key is not None:
                    auth_credentials = weaviate.auth.Auth.api_key(api_key=api_key)
                elif auth_config is not None:
                    assert "client_secret" in auth_config, (
                        "client_secret is required in auth_config"
                    )
                    auth_credentials = weaviate.auth.Auth.client_credentials(
                        client_secret=auth_config.pop("client_secret"), **auth_config
                    )

                # Use provided grpc_host or default to HTTP host
                actual_grpc_host = grpc_host if grpc_host is not None else host

                self.client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=http_secure,
                    grpc_host=actual_grpc_host,
                    grpc_port=grpc_port,
                    grpc_secure=grpc_secure,
                    auth_credentials=auth_credentials,
                    headers=additional_headers,
                )
        else:
            self.client = client

        # Store embedding model
        self.embedding_model = embedding_model

        # Set up batch configuration
        self.batch_size = batch_size
        self.batch_dynamic = batch_dynamic
        self.batch_timeout_retries = batch_timeout_retries

        # Initialize the collection
        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name(sep="_")
                # Check if the collection exists
                if not self._collection_exists(self.collection_name):
                    break
            logger.info(f"Chonkie created a new collection in Weaviate: {self.collection_name}")
        else:
            self.collection_name = collection_name

        # Create the collection if it doesn't exist
        if not self._collection_exists(self.collection_name):
            self._create_collection()

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("weaviate") is not None

    def close(self) -> None:
        """Close."""
        self.client.close()

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global weaviate
            import weaviate
        else:
            raise ImportError("Please install it with `pip install chonkie[weaviate]`.")

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Weaviate.

        Args:
            collection_name: The name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.

        """
        try:
            exists = self.client.collections.exists(collection_name)
            return exists
        except Exception as e:
            logger.warning(f"Failed to check for collection '{collection_name}': {e}")
            return False

    def _create_collection(self) -> None:
        """Create a new collection in Weaviate."""
        from weaviate.collections.classes.config import Configure, DataType, Property

        try:
            # Define the collection schema
            self.client.collections.create(
                name=self.collection_name,
                vector_index_config=Configure.VectorIndex.hnsw(),
                properties=[
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="The text content of the chunk",
                    ),
                    Property(
                        name="start_index",
                        data_type=DataType.INT,
                        description="The start index of the chunk in the original text",
                    ),
                    Property(
                        name="end_index",
                        data_type=DataType.INT,
                        description="The end index of the chunk in the original text",
                    ),
                    Property(
                        name="token_count",
                        data_type=DataType.INT,
                        description="The number of tokens in the chunk",
                    ),
                    Property(
                        name="chunk_type",
                        data_type=DataType.TEXT,
                        description="The type of the chunk",
                    ),
                ],
            )

            logger.info(f"Created Weaviate collection: {self.collection_name}")
        except Exception:
            raise


    def __repr__(self) -> str:
        """Return the string representation of the WeaviateConnection."""
        return f"WeaviateConnection(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query.

        Args:
            query: Optional[str]: The query string to search for.
            embedding: Optional[List[float]]: The embedding vector to search for. If provided, `query` is ignored.
            limit: int: The number of top similar chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: The list of most similar chunks with their metadata.

        """
        logger.debug(f"Searching Weaviate collection: {self.collection_name} with limit={limit}")
        if embedding is None and query is None:
            raise ValueError("Either query or embedding must be provided")
        if query is not None:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            embedding = self.embedding_model.encode(query).tolist()
        collection = self.client.collections.get(self.collection_name)
        # Weaviate expects a vector for similarity search
        results = collection.query.near_vector(
            near_vector=embedding, # type: ignore[arg-type] 
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )
        # Format results to match other handshakes
        matches = []
        for obj in results.objects:
            score = getattr(obj.metadata, "distance", None) if obj.metadata else None
            # Weaviate returns distance, convert to similarity (1 - distance) if needed
            similarity = 1.0 - score if score is not None else None
            match = {
                "id": obj.uuid,
                "score": similarity,
                "text": obj.properties.get("text"),
                "start_index": obj.properties.get("start_index"),
                "end_index": obj.properties.get("end_index"),
                "token_count": obj.properties.get("token_count"),
                "chunk_type": obj.properties.get("chunk_type"),
            }
            matches.append(match)
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
