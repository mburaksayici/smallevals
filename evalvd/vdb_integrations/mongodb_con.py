"""MongoDB Connection to export Chonkie's Chunks into a MongoDB collection."""

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
    import pymongo
    from sentence_transformers import SentenceTransformer


class MongoDBConnection(BaseVDBConnection):
    """MongoDB Connection to export Chonkie's Chunks into a MongoDB collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: The MongoDB client to use. If None, will create a new client.
        uri: The MongoDB connection URI.
        username: MongoDB username for authentication.
        password: MongoDB password for authentication.
        hostname: MongoDB host address.
        port: MongoDB port number.
        db_name: The name of the database or "random" for auto-generated name.
        collection_name: The name of the collection or "random" for auto-generated name.
        embedding_model: The embedding model identifier or instance.
        **kwargs: Additional keyword arguments for MongoDB client.

    """

    def __init__(
        self,
        client: Optional["pymongo.MongoClient"] = None,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[Union[int, str]] = None,
        db_name: Union[str, Literal["random"]] = "random",
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MongoDB Connection with the specified connection parameters.

        Args:
            client: The MongoDB client to use. If None, will create a new client.
            uri: The MongoDB connection URI.
            username: MongoDB username for authentication.
            password: MongoDB password for authentication.
            hostname: MongoDB host address.
            port: MongoDB port number.
            db_name: The name of the database or "random" for auto-generated name.
            collection_name: The name of the collection or "random" for auto-generated name.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            **kwargs: Additional keyword arguments for MongoDB client.

        """
        super().__init__()
        self._import_dependencies()

        if client is not None:
            self.client = client
        else:
            # use uri
            if uri is None:
                # construct the uri
                if hostname is not None:
                    if username is not None and password is not None:
                        uri = f"mongodb://{username}:{password}@{hostname}"
                    else:
                        uri = f"mongodb://{hostname}"
                # use localhost
                else:
                    logger.info("No hostname provided, using localhost instead")
                    port = str(port) if port is not None else "27017"
                    uri = f"mongodb://localhost:{port}"
                    # clear port
                    port = None

            self.client = pymongo.MongoClient(
                uri,
                port=int(port) if port is not None else None,
                **kwargs,
            )

        if db_name == "random":
            self.db_name = generate_random_collection_name()
            logger.info(f"Chonkie created a new MongoDB database: {self.db_name}")
        else:
            self.db_name = db_name
        self.db = self.client[self.db_name]

        if collection_name == "random":
            self.collection_name = generate_random_collection_name()
            logger.info(f"Chonkie created a new MongoDB collection: {self.collection_name}")
        else:
            self.collection_name = collection_name
        self.collection = self.db[self.collection_name]

        # Store embedding model
        self.embedding_model = embedding_model

    def _is_available(self) -> bool:
        return importutil.find_spec("pymongo") is not None

    def _import_dependencies(self) -> None:
        if self._is_available():
            global pymongo
            import pymongo
        else:
            raise ImportError(
                "pymongo is not installed. Please install it with `pip install chonkie[mongodb]`."
            )

    def __repr__(self) -> str:
        """Return a string representation of the MongoDBConnection instance."""
        return f"MongoDBConnection(db_name={self.db_name}, collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in the MongoDB collection.

        Args:
            query: The query string to search for.
            embedding: The embedding vector to search for.
            limit: The number of top similar chunks to return.

        Returns:
            A list of dictionaries containing the similar chunks and their metadata.

        """
        logger.debug(f"Searching MongoDB collection: {self.collection_name} with limit={limit}")
        if query is not None:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            embedding = self.embedding_model.encode(query).tolist()
        if embedding is None:
            raise ValueError("Either query (with embedding_model) or embedding must be provided")
        # Get all documents with embeddings
        docs = list(
            self.collection.find(
                {},
                {
                    "_id": 1,
                    "text": 1,
                    "embedding": 1,
                    "start_index": 1,
                    "end_index": 1,
                    "token_count": 1,
                },
            )
        )

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Compute cosine similarity between two vectors."""
            import math

            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # Score and sort
        results = []
        for doc in docs:
            emb = doc.get("embedding")
            if emb is not None:
                score = cosine_similarity(embedding, emb) # type: ignore[arg-type]
                result = {
                    "id": doc["_id"],
                    "score": score,
                    "text": doc["text"],
                    "start_index": doc.get("start_index"),
                    "end_index": doc.get("end_index"),
                    "token_count": doc.get("token_count"),
                }
                results.append(result)
        # Sort by score descending and return limit
        results.sort(key=lambda x: x["score"], reverse=True)
        matches = results[:limit]
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
