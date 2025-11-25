"""Milvus Connection to export Chonkie's Chunks into a Milvus collection."""

import importlib.util
import os
from pathlib import Path
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
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseVDBConnection
from .utils import generate_random_collection_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
    )
    from sentence_transformers import SentenceTransformer

class MilvusConnection(BaseVDBConnection):
    """Milvus Connection to export Chonkie's Chunks into a Milvus collection.

    This handshake connects to a Milvus instance, creates a collection with a
    defined schema, and ingests chunks for similarity search.

    Args:
        client: An optional pre-initialized MilvusClient instance.
        uri: The URI to connect to Milvus (e.g., "http://localhost:19530").
        collection_name: The name of the collection to use. If "random", a unique name is generated.
        embedding_model: The embedding model to use for vectorizing chunks.
        host: The host of the Milvus instance. Defaults to "localhost".
        port: The port of the Milvus instance. Defaults to "19530".
        user: The username to connect to Milvus. Defaults to "".
        api_key: The API key to connect to Milvus. Defaults to "".
        **kwargs: Additional keyword arguments for future use.
        
    """

    def __init__(
        self,
        client: Optional[Any] = None, 
        uri: Optional[str] = None,
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        dimension: int = 384,
        host: str = "localhost",
        port: str = "19530",
        user: Optional[str] = "",
        api_key: Optional[str] = "",
        alias: str = "default",
        **kwargs: Any,
    ) -> None:
        """Initialize the Milvus Connection.

        Args:
            client: An optional pre-initialized MilvusClient instance.
            uri: The URI to connect to Milvus (e.g., "http://localhost:19530").
            collection_name: The name of the collection to use. If "random", a unique name is generated.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            dimension: The dimension of the vectors. Required when creating a new collection.
            host: The host of the Milvus instance. Defaults to "localhost".
            port: The port of the Milvus instance. Defaults to "19530".
            user: The username to connect to Milvus. Defaults to "".
            api_key: The API key to connect to Milvus. Defaults to "".
            alias: The alias to use for the Milvus connection. Defaults to "default".
            **kwargs: Additional keyword arguments for future use.

        """
        self._import_dependencies()
        super().__init__()
        self.alias = alias

        # Check if uri is a local folder path for Milvus Lite
        use_local_storage = False
        local_db_path = None
        
        if uri is not None:
            # Check if uri is a folder path (local file-based storage)
            uri_str = str(uri)
            uri_path = Path(uri_str)
            if uri_path.exists() and uri_path.is_dir():
                use_local_storage = True
                local_db_path = str(uri_path.resolve())
                logger.info(f"Using local file-based Milvus storage at: {local_db_path}")
            elif not uri_str.startswith(('http://', 'https://', 'file://')):
                # Might be a path that doesn't exist yet - create it
                try:
                    uri_path.mkdir(parents=True, exist_ok=True)
                    if uri_path.exists() and uri_path.is_dir():
                        use_local_storage = True
                        local_db_path = str(uri_path.resolve())
                        logger.info(f"Created local Milvus storage directory: {local_db_path}")
                except Exception:
                    # Not a valid path, treat as URI
                    pass

        # 1. Establish connection using the ORM's global connection manager
        if client is not None:
            self.client = client
        elif use_local_storage and local_db_path:
            # Use Milvus Lite with local file storage
            # For Milvus Lite, use file:// URI or pass data_path
            self.client = MilvusClient(uri=f"file://{local_db_path}", **kwargs) # type: ignore
            try:
                connections.connect(uri=f"file://{local_db_path}", alias=alias, **kwargs) # type: ignore
            except Exception as e:
                logger.warning(f"Could not connect with ORM connections: {e}")
        else:
            # Use server-based connection
            self.client = MilvusClient(uri=uri, host=host, port=port, user=user, password=api_key, alias=alias, **kwargs) # type: ignore
            # Always connect using ORM before any collection operations
            try:
                connections.connect(uri=uri, host=host, port=port, user=user, password=api_key, alias=alias, **kwargs) # type: ignore
            except Exception as e:
                logger.warning(f"Could not connect with ORM connections: {e}")
        # 3. Store embedding model and dimension
        self.embedding_model = embedding_model
        self.dimension = dimension

        # 4. Handle collection name and schema

        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name(sep="_")
                # Pass alias explicitly to utility.has_collection
                if not self.client.has_collection(self.collection_name): # type: ignore
                    break
        else:
            self.collection_name = collection_name

        # Pass alias explicitly to utility.has_collection
        if not self.client.has_collection(self.collection_name):
            self._create_collection_with_schema()

        self.collection = Collection(self.collection_name) # type: ignore
        self.collection.load()

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Collection, CollectionSchema, DataType, FieldSchema, connections, utility, ConnectionNotExistException, MilvusClient
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                MilvusClient,
                connections,
                utility,
            )
            from pymilvus.exceptions import ConnectionNotExistException
        else:
            raise ImportError(
                "Milvus is not installed. "
                + "Please install it with `pip install pymilvus`."
            )

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importlib.util.find_spec("pymilvus") is not None

    def _create_collection_with_schema(self) -> None:
        """Create a new collection with a predefined schema and index."""
        # Define fields: pk, text, metadata, and the vector embedding
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True), # type: ignore
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535), # type: ignore
            FieldSchema(name="start_index", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="end_index", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="token_count", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension), # type: ignore
        ]
        schema = CollectionSchema(fields, description="Chonkie Connection Collection") # type: ignore
        collection = Collection(self.collection_name, schema) # type: ignore
        logger.info(f"Chonkie created a new collection in Milvus: {self.collection_name}")

        # Create a default index for the vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Created default HNSW index on 'embedding' field.")

    def __repr__(self) -> str:
        """Return the string representation of the MilvusConnection."""
        return f"MilvusConnection(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], "np.ndarray"]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query."""
        if embedding is None and query is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        if query:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            query_embedding = self.embedding_model.encode(query)
            # Milvus expects a list of vectors for searching
            query_vectors = [query_embedding.tolist()]
        else:
            # Ensure embedding is in the correct format (list of lists)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            # If it's a flat list, wrap it in another list
            if embedding and len(embedding) > 0 and isinstance(embedding[0], float):
                query_vectors = [embedding]
            else:
                query_vectors = embedding # type: ignore

        # Default search parameters for HNSW index
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        output_fields = ["text", "start_index", "end_index", "token_count"]

        results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields,
        )

        # Format results into a standardized list of dicts
        matches = []
        # Results are for the first query vector (index 0)
        for hit in results[0]:
            match_data = {
                "id": hit.id,
                "score": hit.distance, # Milvus uses 'distance', which is analogous to score
                **hit.entity,
            }
            matches.append(match_data)
        return matches