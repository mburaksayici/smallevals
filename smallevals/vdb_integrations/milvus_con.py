"""Milvus Connection to export Chonkie's Chunks into a Milvus collection."""

import importlib.util
import logging
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
        MilvusClient,
    )
    from sentence_transformers import SentenceTransformer


class MilvusConnection(BaseVDBConnection):
    """Milvus Connection to.

    This connects to a Milvus instance, creates/uses a collection with a
    defined schema, and enables similarity search over chunks.

    Two common ways to use:

    1) With an existing Collection (e.g. from tests):

        from pymilvus import Collection
        col = Collection("TestCollection", using="test_connection")
        conn = MilvusConnection(client=col, embedding_model=embedding_model)

    2) With a running Milvus server:

        conn = MilvusConnection(
            uri="http://localhost:19530",
            collection_name="my_collection",
            embedding_model=embedding_model,
            dimension=384,
        )
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
        self._import_dependencies()
        super().__init__()

        self.alias = alias
        self.embedding_model = embedding_model
        self.dimension = dimension



        # --- Special case: user passed an existing Collection directly ---
        from pymilvus import Collection as _Collection  # type: ignore
        from pymilvus import MilvusClient as _MilvusClient  # type: ignore

        if isinstance(client, _Collection):
            # We already have a live collection, just use it.
            self.client = None
            self.collection = client
            self.collection_name = client.name
            logger.info(f"Using provided Milvus Collection: {self.collection_name}")
            # Ensure it's loaded for search
            try:
                self.collection.load()
                self.anns_field = next(
            f.name for f in self.collection.schema.fields 
            if f.dtype.name == "FLOAT_VECTOR"
        )


            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load collection {self.collection_name}: {e}")
            return

        # --- Detect local Milvus Lite (file-based) vs server ---
        use_local_storage = False
        local_db_path: Optional[str] = None

        if uri is not None:
            uri_str = str(uri)
            uri_path = Path(uri_str)
            if uri_path.exists() and uri_path.is_dir():
                use_local_storage = True
                local_db_path = str(uri_path.resolve())
                logger.info(f"Using local file-based Milvus storage at: {local_db_path}")
            elif not uri_str.startswith(("http://", "https://", "file://")):
                # Might be a folder path that doesn't exist yet
                try:
                    uri_path.mkdir(parents=True, exist_ok=True)
                    if uri_path.exists() and uri_path.is_dir():
                        use_local_storage = True
                        local_db_path = str(uri_path.resolve())
                        logger.info(f"Created local Milvus storage directory: {local_db_path}")
                except Exception:
                    # Not a valid path, treat as normal URI
                    pass

        # --- Establish client + ORM connection ---
        if isinstance(client, _MilvusClient):
            # User provided a MilvusClient directly
            self.client = client
            self.anns_field = next(
                f.name for f in self.collection.schema.fields 
                if f.dtype.name == "FLOAT_VECTOR"
            )
        elif use_local_storage and local_db_path:
            # Milvus Lite - file-based
            from pymilvus import MilvusClient, connections  # type: ignore

            self.client = MilvusClient(uri=f"file://{local_db_path}", **kwargs)
            try:
                connections.connect(uri=f"file://{local_db_path}", alias=alias, **kwargs)
                self.anns_field = next(
                    f.name for f in self.collection.schema.fields 
                    if f.dtype.name == "FLOAT_VECTOR"
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not connect ORM (Lite) with alias={alias}: {e}")
        else:
            # Normal server-based Milvus
            from pymilvus import MilvusClient, connections  # type: ignore

            # If uri not provided, construct from host/port
            final_uri = uri or f"http://{host}:{port}"
            self.client = MilvusClient(
                uri=final_uri,
                host=host,
                port=port,
                user=user,
                password=api_key,
                **kwargs,
            )
            try:

                connections.connect(
                    alias=alias,
                    uri=final_uri,
                    user=user,
                    password=api_key,
                    **kwargs,
                )
                self.anns_field = next(
                    f.name for f in self.collection.schema.fields 
                    if f.dtype.name == "FLOAT_VECTOR"
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not connect ORM with alias={alias}: {e}")


    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Collection, CollectionSchema, DataType, FieldSchema
            global connections, utility, ConnectionNotExistException, MilvusClient

            from pymilvus import (  # type: ignore
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                MilvusClient,
                connections,
                utility,
            )
            from pymilvus.exceptions import ConnectionNotExistException  # type: ignore
        else:
            raise ImportError(
                "Milvus is not installed. "
                "Please install it with `pip install pymilvus`."
            )

    def _is_available(self) -> bool:
        """Check if pymilvus is installed."""
        return importlib.util.find_spec("pymilvus") is not None

    def _has_collection(self, name: str) -> bool:
        """Check if the collection exists, using client or utility."""
        # Prefer MilvusClient.has_collection if available
        if getattr(self, "client", None) is not None and hasattr(self.client, "has_collection"):
            try:
                return bool(self.client.has_collection(name))  # type: ignore
            except Exception:
                pass

        # Fallback to ORM utility
        try:
            return bool(utility.has_collection(name, using=self.alias))  # type: ignore
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"MilvusConnection(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], "np.ndarray"]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar chunks to the query."""
        if embedding is None and query is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        # Encode query text if needed
        if query:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            query_embedding = self.embedding_model.encode(query)
            query_vectors = [query_embedding.tolist()]
        else:
            # Ensure embedding is list-of-lists
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            if embedding and len(embedding) > 0 and isinstance(embedding[0], float):
                query_vectors = [embedding]  # type: ignore[list-item]
            else:
                query_vectors = embedding  # type: ignore[assignment]

        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        output_fields = ["text", "start_index", "end_index", "token_count"]

        results = self.collection.search(
            data=query_vectors,
            anns_field=self.anns_field,
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        matches: List[Dict[str, Any]] = []
        for hit in results[0]:
            # hit.entity is a dict-like object with fields
            entity_data = dict(hit.entity)
            match_data = {
                "id": str(hit.id),
                "score": hit.distance,  # distance as score
                **entity_data,
            }
            matches.append(match_data)

        return matches

    def sample_chunks(self, num_chunks: int = 20) -> List[Dict[str, Any]]:
        """
        Randomly sample k chunks from Milvus using ID-first strategy (fast & safe).

        Returns:
            List of dicts with chunk info.
        """
        import random
        # --- 1. Fetch all primary keys only ---
        ids = self.collection.query(
            expr="",
            output_fields=["pk"],
            limit=num_chunks*2,
            filter="RANDOM_SAMPLE(0.99)",
        )
        ids =  [i["pk"] for i in ids]


        if not ids:
            return []

        # --- 2. Sample IDs in Python ---
        sampled_ids = random.sample(ids, num_chunks)

        # --- 3. Fetch full rows only for sampled IDs ---
        # Build Milvus IN expression
        id_expr = f"pk in {sampled_ids}"

        results = self.collection.query(
            expr=id_expr,
            output_fields=[
                "pk",
                "text",
                "start_index",
                "end_index",
                "token_count"
            ]
        )

        # --- 4. Normalize output format ---
        chunks = []
        for row in results:
            chunks.append({
                "id": str(row.get("pk")),
                "text": row.get("text", ""),
                "start_index": row.get("start_index"),
                "end_index": row.get("end_index"),
                "token_count": row.get("token_count"),
            })

        return chunks

