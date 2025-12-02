"""Pgvector Connection using SQLAlchemy or psycopg2 for PostgreSQL with pgvector extension."""

import importlib.util as importutil
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import json

from .base import BaseVDBConnection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class PgvectorConnection(BaseVDBConnection):
    """Pgvector Connection for PostgreSQL database with pgvector extension.
    
    This connection supports both SQLAlchemy (Engine, Connection, Session) and 
    psycopg2 connections for storing and querying vector embeddings using 
    PostgreSQL's pgvector extension.

    Args:
        client: A SQLAlchemy Engine/Connection/Session or psycopg2 connection object.
        collection_name: The name of the table to store vectors in.
        embedding_model: The embedding model to use for generating embeddings.
        vector_dimensions: The number of dimensions for the vector embeddings.
        metric: Distance metric to use ('l2', 'cosine', 'inner_product'). Defaults to 'l2'.

    """

    def __init__(
        self,
        client: Any,
        collection_name: str = "vectors",
        embedding_model: Optional["SentenceTransformer"] = None,
        vector_dimensions: Optional[int] = None,
        metric: str = "l2",
    ) -> None:
        """Initialize the Pgvector Connection.
        
        Args:
            client: SQLAlchemy Engine/Connection/Session or psycopg2 connection.
            collection_name: The name of the table to store vectors in.
            embedding_model: Optional SentenceTransformer model. If provided, used for query encoding.
            vector_dimensions: The number of dimensions for vectors. Required if embedding_model not provided.
            metric: Distance metric ('l2', 'cosine', 'inner_product'). Defaults to 'l2'.

        """         
        super().__init__()
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.metric = metric
        
        # Determine vector dimensions
        if vector_dimensions is not None:
            self.vector_dimensions = vector_dimensions
        elif embedding_model is not None:
            self.vector_dimensions = embedding_model.get_sentence_embedding_dimension()
        else:
            raise ValueError("Either embedding_model or vector_dimensions must be provided")
        
        # Detect connection type and setup
        self._detect_and_setup_connection(client)
        
        # Enable pgvector extension and create table
        self._setup_database()

    def _detect_and_setup_connection(self, client: Any) -> None:
        """Detect whether client is SQLAlchemy or psycopg2 and setup accordingly."""
        client_class = client.__class__.__name__
        client_module = client.__class__.__module__
        
        # Check if it's SQLAlchemy
        if "sqlalchemy" in client_module:
            self.connection_type = "sqlalchemy"
            self.client = client
            
            # Determine if it's Engine, Connection, or Session
            if "Engine" in client_class:
                self.sqlalchemy_type = "engine"
            elif "Connection" in client_class:
                self.sqlalchemy_type = "connection"
            elif "Session" in client_class:
                self.sqlalchemy_type = "session"
            else:
                self.sqlalchemy_type = "unknown"
                
            logger.info(f"Detected SQLAlchemy {self.sqlalchemy_type}")
            
        # Check if it's psycopg2
        elif "psycopg2" in client_module or client_class == "connection":
            self.connection_type = "psycopg2"
            self.client = client
            logger.info("Detected psycopg2 connection")
            
        else:
            raise ValueError(
                f"Unsupported connection type: {client_class}. "
                "Expected SQLAlchemy Engine/Connection/Session or psycopg2 connection"
            )

    def _execute_sql(self, sql: str, params: Optional[tuple] = None, fetch: bool = False) -> Any:
        """Execute SQL query using the appropriate connection type."""
        if self.connection_type == "sqlalchemy":
            from sqlalchemy import text
            sql_text = text(sql)
            
            if self.sqlalchemy_type == "engine":
                with self.client.connect() as conn:
                    result = conn.execute(sql_text, params or {})
                    if fetch:
                        fetched = result.fetchall()
                        conn.commit()
                        return fetched
                    conn.commit()
                    return result
            elif self.sqlalchemy_type == "connection":
                result = self.client.execute(sql_text, params or {})
                if fetch:
                    return result.fetchall()
                self.client.commit()
                return result
            elif self.sqlalchemy_type == "session":
                result = self.client.execute(sql_text, params or {})
                if fetch:
                    fetched = result.fetchall()
                    self.client.commit()
                    return fetched
                self.client.commit()
                return result
                
        elif self.connection_type == "psycopg2":
            cursor = self.client.cursor()
            try:
                cursor.execute(sql, params or ())
                if fetch:
                    result = cursor.fetchall()
                    cursor.close()
                    return result
                self.client.commit()
                cursor.close()
            except Exception as e:
                self.client.rollback()
                cursor.close()
                raise e

    def _setup_database(self) -> None:
        """Enable pgvector extension and create table if not exists."""
        # Enable pgvector extension
        try:
            self._execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
        
        # Create table with vector column and metadata
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.collection_name} (
            id TEXT PRIMARY KEY,
            embedding vector({self.vector_dimensions}),
            metadata JSONB,
            text TEXT,
            start_index INTEGER,
            end_index INTEGER,
            token_count INTEGER
        );
        """
        
        try:
            self._execute_sql(create_table_sql)
            logger.info(f"Table '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    def _get_distance_operator(self) -> str:
        """Get the pgvector distance operator based on metric."""
        operators = {
            "l2": "<->",           # L2 distance (Euclidean)
            "cosine": "<=>",       # Cosine distance
            "inner_product": "<#>" # Inner product (negative, so larger is closer)
        }
        return operators.get(self.metric, "<->")

    def search(
        self, 
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_value: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using pgvector similarity.
        
        Args:
            query: The query text to search for.
            query_embedding: Pre-computed query embedding vector.
            limit: Maximum number of results to return.
            filters: Optional metadata filters (not fully implemented yet).
            include_metadata: Whether to include metadata in results.
            include_value: Whether to include similarity scores in results.
            
        Returns:
            List[Dict[str, Any]]: List of similar vectors with metadata and scores.

        """
        logger.debug(f"Searching PostgreSQL table: {self.collection_name} with limit={limit}")
        
        # Determine the query embedding
        if query_embedding is None:
            if query is None:
                raise ValueError("Either query or query_embedding must be provided")
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            query_embedding = self.embedding_model.encode(query).tolist()
        
        # Convert embedding to PostgreSQL vector format
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Build SQL query
        distance_op = self._get_distance_operator()
        
        select_fields = ["id"]
        if include_value:
            select_fields.append(f"embedding {distance_op} %(embedding)s::vector AS distance")
        if include_metadata:
            select_fields.extend(["metadata", "text", "start_index", "end_index", "token_count"])
        
        sql = f"""
        SELECT {", ".join(select_fields)}
        FROM {self.collection_name}
        ORDER BY embedding {distance_op} %(embedding)s::vector
        LIMIT %(limit)s;
        """
        
        # Execute query
        if self.connection_type == "sqlalchemy":
            # SQLAlchemy needs text() wrapper for raw SQL
            try:
                from sqlalchemy import text
                # Convert psycopg2-style placeholders into SQLAlchemy style and fix pgvector casting
                sql_sqlalchemy = (
                    sql
                    .replace("%(limit)s", ":limit")
                    .replace("%(embedding)s", "CAST(:embedding AS vector)")
                )

                sql_text = text(sql_sqlalchemy)

                params = {
                    "embedding": embedding_str,   # MUST be '[0.1, 0.2, ...]' format string
                    "limit": limit,
                }

                if self.sqlalchemy_type == "engine":
                    with self.client.connect() as conn:
                        result = conn.execute(sql_text, params)
                        rows = result.fetchall()

                elif self.sqlalchemy_type == "connection":
                    result = self.client.execute(sql_text, params)
                    rows = result.fetchall()

                else:  # session
                    result = self.client.execute(sql_text, params)
                    rows = result.fetchall()


            except Exception as e:
                logger.error(f"SQLAlchemy query error: {e}")
                raise
                
        else:  # psycopg2
            cursor = self.client.cursor()
            try:
                cursor.execute(sql, {"embedding": embedding_str, "limit": limit})
                rows = cursor.fetchall()
                cursor.close()
            except Exception as e:
                cursor.close()
                logger.error(f"psycopg2 query error: {e}")
                raise
        
        # Format results
        formatted_results = []
        for row in rows:
            idx = 0
            result_dict = {"id": row[idx]}
            idx += 1
            
            if include_value:
                result_dict["similarity"] = float(row[idx])
                idx += 1
            
            if include_metadata:
                metadata_json = row[idx]
                if metadata_json:
                    try:
                        result_dict["metadata"] = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                    except:
                        result_dict["metadata"] = metadata_json
                idx += 1
                
                result_dict["text"] = row[idx]
                idx += 1
                result_dict["start_index"] = row[idx]
                idx += 1
                result_dict["end_index"] = row[idx]
                idx += 1
                result_dict["token_count"] = row[idx]
        
            formatted_results.append(result_dict)
        
        logger.info(f"Search complete: found {len(formatted_results)} matching vectors")
        return formatted_results

    def create_index(self, method: str = "hnsw", **index_params: Any) -> None:
        """Create a vector index for improved search performance.
        
        Args:
            method: Index method ('hnsw' or 'ivfflat'). Defaults to 'hnsw'.
            **index_params: Additional parameters:
                - m: Max connections per layer for HNSW (default: 16)
                - ef_construction: Size of dynamic candidate list for HNSW (default: 64)
                - lists: Number of inverted lists for IVFFlat (default: 100)

        """
        distance_op = self._get_distance_operator()
        index_name = f"{self.collection_name}_embedding_idx"
        
        # Drop existing index if exists
        drop_sql = f"DROP INDEX IF EXISTS {index_name};"
        self._execute_sql(drop_sql)
        
        if method.lower() == "hnsw":
            m = index_params.get("m", 16)
            ef_construction = index_params.get("ef_construction", 64)
            
            create_index_sql = f"""
            CREATE INDEX {index_name} ON {self.collection_name}
            USING hnsw (embedding vector_{self.metric}_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
            """
            
        elif method.lower() == "ivfflat":
            lists = index_params.get("lists", 100)
            
            create_index_sql = f"""
            CREATE INDEX {index_name} ON {self.collection_name}
            USING ivfflat (embedding vector_{self.metric}_ops)
            WITH (lists = {lists});
            """
        else:
            raise ValueError(f"Unsupported index method: {method}. Use 'hnsw' or 'ivfflat'")
        
        self._execute_sql(create_index_sql)
        logger.info(f"Created {method} index on table: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the table/collection."""
        # Query table info
        sql = f"""
        SELECT COUNT(*) as count FROM {self.collection_name};
        """
        
        result = self._execute_sql(sql, fetch=True)
        count = result[0][0] if result else 0
        
        return {
            "name": self.collection_name,
            "dimension": self.vector_dimensions,
            "metric": self.metric,
            "count": count,
        }

    def __repr__(self) -> str:
        """Return the string representation of the PgvectorConnection."""
        return f"PgvectorConnection(collection_name={self.collection_name}, vector_dimensions={self.vector_dimensions}, type={self.connection_type})"
