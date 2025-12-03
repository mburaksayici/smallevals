"""SmallEvals VDB Connection wrapper that auto-detects vector database types."""

import logging
from typing import Any, Optional, Union

from .base import BaseVDBConnection

logger = logging.getLogger(__name__)


class SmallEvalsVDBConnection:
    """Wrapper that auto-detects vector database type and creates appropriate connection.
    
    This class automatically detects the type of vector database from the connection object
    and wraps it with the appropriate BaseVDBConnection subclass.
    
    Args:
        connection: Raw connection object (e.g., chromadb.Client, pinecone.Pinecone, etc.)
        collection: Collection/index name
        embedding: SentenceTransformer instance or HuggingFace model name string
    
    Example:
        >>> import chromadb
        >>> from sentence_transformers import SentenceTransformer
        >>> embedding = SentenceTransformer("intfloat/e5-small-v2")
        >>> client = chromadb.PersistentClient(path="./chromadb")
        >>> vdb = SmallEvalsVDBConnection(
        ...     connection=client,
        ...     collection="my_collection",
        ...     embedding=embedding
        ... )
    """
    
    def __init__(
        self,
        connection: Any,
        collection: str,
        embedding: Union[Any, str],
        *args,
        **kwargs
    ):
        """Initialize SmallEvalsVDBConnection with auto-detection."""

        self.raw_connection = connection
        self.collection_name = collection
        self._embedding = embedding
        
        # Load embedding model if string provided
        if isinstance(embedding, str):
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model from HuggingFace: {embedding}")
                self.embedding_model = SentenceTransformer(embedding)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required when embedding is a string. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            self.embedding_model = embedding
        
        # Detect VDB type before creating connection
        self.vdb_type = self._detect_connection_type(connection)
        logger.info(f"Detected VDB type: {self.vdb_type}")
        
        # Auto-detect and create appropriate connection
        self.connection = self._auto_detect_and_create(connection, collection, *args, **kwargs)
    
    def _auto_detect_and_create(
        self, connection: Any, collection: str, *args, **kwargs
    ) -> BaseVDBConnection:
        """Auto-detect vector database type and create appropriate connection."""
        connection_type = self.vdb_type
        
        if connection_type == "chromadb":
            return self._create_chroma_connection(connection, collection, *args, **kwargs)
        elif connection_type == "pinecone":
            return self._create_pinecone_connection(connection, collection, *args, **kwargs)
        elif connection_type == "milvus":
            return self._create_milvus_connection(connection, collection, *args, **kwargs)
        elif connection_type == "qdrant":
            return self._create_qdrant_connection(connection, collection, *args, **kwargs)
        elif connection_type == "weaviate":
            return self._create_weaviate_connection(connection, collection, *args, **kwargs)
        elif connection_type == "elastic":
            return self._create_elastic_connection(connection, collection, *args, **kwargs)
        elif connection_type == "mongodb":
            return self._create_mongodb_connection(connection, collection, *args, **kwargs)
        elif connection_type == "pgvector":
            return self._create_pgvector_connection(connection, collection, *args, **kwargs)
        elif connection_type == "turbopuffer":
            return self._create_turbopuffer_connection(connection, collection, *args, **kwargs)
        elif connection_type == "faiss":
            return self._create_faiss_connection(connection, collection, *args, **kwargs)
        else:
            raise ValueError(
                f"Could not detect vector database type from connection object. "
                f"Supported types: ChromaDB, Pinecone, Milvus, Qdrant, Weaviate, "
                f"Elasticsearch, MongoDB, PgVector, Turbopuffer, FAISS"
            )
    
    def _detect_connection_type(self, connection: Any) -> str:
        """Detect the type of vector database from connection object."""
        connection_class_name = connection.__class__.__name__
        module_name = connection.__class__.__module__

        # Check FAISS first (it's a direct FaissConnection instance)
        if "faiss" in module_name.lower() or connection_class_name == "FaissConnection":
            return "faiss"

        # Check ChromaDB
        if "chromadb" in module_name.lower() or connection_class_name in ["Client", "PersistentClient", "HttpClient"]:
            # Additional check: ChromaDB clients have specific attributes
            if hasattr(connection, "get_or_create_collection") or hasattr(connection, "list_collections"):
                return "chromadb"
        
        # Check Pinecone
        if "pinecone" in module_name.lower() or connection_class_name == "Pinecone":
            return "pinecone"
        
        # Check Milvus
        if "milvus" in module_name.lower() or connection_class_name == "MilvusClient":
            return "milvus"
        
        # Check Qdrant
        if "qdrant" in module_name.lower() or connection_class_name == "QdrantClient":
            return "qdrant"
        
        # Check Weaviate
        if "weaviate" in module_name.lower():
            # Additional check for Weaviate
            return "weaviate"
        
        # Check Elasticsearch
        if "elasticsearch" in module_name.lower() or connection_class_name in ["Elasticsearch", "Elastic"]:
            return "elastic"
        
        # Check MongoDB
        if "mongo" in module_name.lower() or connection_class_name in ["MongoClient", "Database"]:
            return "mongodb"
        
        # Check PgVector (PostgreSQL with SQLAlchemy or psycopg2)
        # Check for SQLAlchemy Engine, Connection, or Session
        if "sqlalchemy" in module_name.lower():
            if "Engine" in connection_class_name or "Connection" in connection_class_name or "Session" in connection_class_name:
                # Verify it's PostgreSQL by checking dialect if possible
                try:
                    if hasattr(connection, "dialect"):
                        if "postgres" in connection.dialect.name.lower():
                            return "pgvector"
                    elif hasattr(connection, "engine") and hasattr(connection.engine, "dialect"):
                        if "postgres" in connection.engine.dialect.name.lower():
                            return "pgvector"
                    else:
                        # Assume PostgreSQL if using SQLAlchemy but can't verify
                        return "pgvector"
                except:
                    # If we can't check, assume PostgreSQL
                    return "pgvector"
        
        # Check for psycopg2 connection
        if "psycopg2" in module_name.lower() or (connection_class_name == "connection" and "psycopg" in str(type(connection)).lower()):
            return "pgvector"
        
        # Check Turbopuffer
        if "turbopuffer" in module_name.lower() or connection_class_name == "Client":
            return "turbopuffer"
        
        # Try to infer from common attributes
        if hasattr(connection, "list_collections") and hasattr(connection, "get_or_create_collection"):
            return "chromadb"
        elif hasattr(connection, "describe_index") or hasattr(connection, "Index"):
            return "pinecone"
        elif hasattr(connection, "has_collection"):
            return "milvus"
        elif hasattr(connection, "collection_exists"):
            return "qdrant"
        
        return "unknown"
    
    def _create_chroma_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create ChromaDB connection."""
        from .chroma_con import ChromaConnection
        return ChromaConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_pinecone_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Pinecone connection."""
        from .pinecone_con import PineconeConnection
        return PineconeConnection(
            client=connection,
            index_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_milvus_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Milvus connection."""
        from .milvus_con import MilvusConnection
        
        # Try to get dimension from embedding model
        dimension = None
        if self.embedding_model:
            try:
                dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                pass
        
        return MilvusConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_qdrant_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Qdrant connection."""
        from .qdrant_con import QdrantConnection
        return QdrantConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_weaviate_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Weaviate connection."""
        from .weaviate_con import WeaviateConnection
        return WeaviateConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_elastic_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Elasticsearch connection."""
        from .elastic_con import ElasticConnection
        
        # Try to get dimension from embedding model
        dimension = None
        if self.embedding_model:
            try:
                dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                pass
        
        
        return ElasticConnection(
            client=connection,
            index_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_mongodb_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create MongoDB connection."""
        from .mongodb_con import MongoDBConnection
        return MongoDBConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_pgvector_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create PgVector connection."""
        from .pgvector_con import PgvectorConnection
        
        # Try to get dimension from embedding model
        dimension = None
        if self.embedding_model:
            try:
                dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                pass
        
        # Allow vector_dimensions override from kwargs
        
        return PgvectorConnection(
            client=connection,
            collection_name=collection,
            embedding_model=self.embedding_model,
            vector_dimensions=dimension,
            **kwargs
        )
    
    def _create_turbopuffer_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create Turbopuffer connection."""
        from .turbopuffer_con import TurbopufferConnection
        return TurbopufferConnection(
            namespace=connection,  # Turbopuffer uses namespace directly
            namespace_name=collection,
            embedding_model=self.embedding_model,
            **kwargs
        )
    
    def _create_faiss_connection(self, connection: Any, collection: str, *args, **kwargs) -> BaseVDBConnection:
        """Create FAISS connection."""
        from .faiss_con import FaissConnection
        
        # If connection is already a FaissConnection, just return it
        if isinstance(connection, FaissConnection):
            return connection
        
        # Otherwise, we can't create a FAISS connection from a raw object
        # FAISS doesn't have a client-server model, so connection should already be FaissConnection
        raise ValueError(
            "FAISS connection must be a FaissConnection instance. "
            "Create it first: FaissConnection(embedding_model=model, dimension=384)"
        )
    
    # Delegate all BaseVDBConnection methods to the wrapped connection
    def search(self, query: Optional[str] = None, embedding: Optional[list] = None, top_k: int = 5):
        """Search the vector database."""
        return self.connection.search(query=query, embedding=embedding, top_k=top_k)
    
    
    def sample_chunks(self, num_chunks: int):
        """Sample chunks from the vector database."""
        return self.connection.sample_chunks(num_chunks=num_chunks)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"SmallEvalsVDBConnection(collection={self.collection_name}, type={type(self.connection).__name__})"

