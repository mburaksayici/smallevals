"""Qdrant Connection to export Chonkie's Chunks into a Qdrant collection."""

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
from uuid import NAMESPACE_OID, uuid5

from .base import BaseVDBConnection
from .utils import generate_random_collection_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import qdrant_client
    from qdrant_client.http.models import PointStruct

    try:
        from qdrant_client.http.models import Distance, VectorParams
    except ImportError:

        class VectorParams:  # type: ignore
            """Stub class for qdrant_client VectorParams when not available."""

            pass

        class Distance:  # type: ignore
            """Stub class for qdrant_client Distance when not available."""

            pass

        from sentence_transformers import SentenceTransformer


class QdrantConnection(BaseVDBConnection):
    """Qdrant Connection to export Chonkie's Chunks into a Qdrant collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: Optional[qdrant_client.QdrantClient]: The Qdrant client to use.
        collection_name: Union[str, Literal["random"]]: The name of the collection to use.
        embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
        url: Optional[str]: The URL to the Qdrant Server.
        api_key: Optional[str]: The API key to the Qdrant Server. Only needed for Qdrant Cloud.
        path: Optional[str]: The path to the Qdrant collection locally. If not provided, will create an ephemeral collection.

    """

    def __init__(
        self,
        client: Optional["qdrant_client.QdrantClient"] = None,
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        dimension: Optional[int] = None,
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Qdrant Connection.

        Args:
            client: Optional[qdrant_client.QdrantClient]: The Qdrant client to use.
            collection_name: Union[str, Literal["random"]]: The name of the collection to use.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            dimension: Optional[int]: The dimension of the vectors. Required if embedding_model is not provided.
            url: Optional[str]: The URL to the Qdrant Server.
            path: Optional[str]: The path to the Qdrant collection locally. If not provided, will create an ephemeral collection.
            api_key: Optional[str]: The API key to the Qdrant Server. Only needed for Qdrant Cloud.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.

        """
        super().__init__()

        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize the Qdrant client
        if client is None:
            if url is not None and api_key is not None:
                self.client = qdrant_client.QdrantClient(
                    url=url,
                    api_key=api_key,
                    **kwargs,  # type: ignore[arg-type]
                )
            elif url is not None:
                self.client = qdrant_client.QdrantClient(url=url, **kwargs)  # type: ignore[arg-type]
            elif path is not None:
                self.client = qdrant_client.QdrantClient(path=path, **kwargs)  # type: ignore[arg-type]
            else:
                # If no client is provided, create an ephemeral collection
                self.client = qdrant_client.QdrantClient(":memory:", **kwargs)  # type: ignore[arg-type]
        else:
            self.client = client

        # Store embedding model and determine dimension
        self.embedding_model = embedding_model
        if dimension is not None:
            self.dimension = dimension
        elif embedding_model is not None:
            self.dimension = embedding_model.get_sentence_embedding_dimension()
        else:
            raise ValueError("Either embedding_model or dimension must be provided")

        # Initialize the collection
        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name()
                # Check if the collection exists or not?
                if not self.client.collection_exists(self.collection_name):
                    break
                else:
                    pass
            logger.info(f"Chonkie created a new collection in Qdrant: {self.collection_name}")
        else:
            self.collection_name = collection_name

        # Create the collection, if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importutil.find_spec("qdrant_client") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global qdrant_client, PointStruct, VectorParams, Distance
            import qdrant_client
            from qdrant_client.http.models import PointStruct
            from qdrant_client.models import Distance, VectorParams
        else:
            raise ImportError(
                "Qdrant is not installed. "
                + "Please install it with `pip install chonkie[qdrant]`."
            )




    def __repr__(self) -> str:
        """Return the string representation of the QdrantConnection."""
        return f"QdrantConnection(collection_name={self.collection_name})"

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
        logger.debug(f"Searching Qdrant collection: {self.collection_name} with limit={limit}")
        if embedding is None and query is None:
            raise ValueError("Either query or embedding must be provided")
        if query is not None:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            embedding = self.embedding_model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        )
        matches = [
            {"id": result["id"], "score": result["score"], **result["payload"]}
            for result in results.dict()["points"]
        ]
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
