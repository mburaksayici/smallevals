"""Elasticsearch Connection to export Chonkie's Chunks into an Elasticsearch index."""

import importlib.util
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
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    from sentence_transformers import SentenceTransformer


class ElasticConnection(BaseVDBConnection):
    """Elasticsearch Connection to export Chonkie's Chunks into an Elasticsearch index.

    This handshake connects to an Elasticsearch instance, creates an index with the
    appropriate vector mapping, and ingests chunks for similarity search.

    Args:
        client: Optional[Elasticsearch]: An existing Elasticsearch client instance. If not provided, one will be created.
        index_name: Union[str, Literal["random"]]: The name of the index to use. If "random", a unique name is generated.
        embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
        dimension: The dimension of the vectors. Required when creating a new index.
        hosts: Optional[Union[str, List[str]]]: URL(s) of the Elasticsearch instance(s).
        cloud_id: Optional[str]: The Cloud ID for connecting to an Elastic Cloud deployment.
        api_key: Optional[str]: The API key for authenticating with an Elastic Cloud deployment.
        **kwargs: Additional keyword arguments to pass to the Elasticsearch client constructor.

    """

    def __init__(
        self,
        client: Optional["Elasticsearch"] = None,
        index_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        dimension: int = 384,
        hosts: Optional[Union[str, List[str]]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Elasticsearch Connection."""
        super().__init__()
        self._import_dependencies()

        # 1. Initialize the Elasticsearch client
        if client:
            self.client = client
        elif cloud_id and api_key:
            self.client = Elasticsearch(cloud_id=cloud_id, api_key=api_key, **kwargs) # type: ignore
        elif hosts:
            self.client = Elasticsearch(hosts=hosts, api_key=api_key, **kwargs) # type: ignore
        else:
            # Default to a standard local client if no other connection info is provided
            self.client = Elasticsearch("http://localhost:9200", **kwargs) # type: ignore

        # 2. Store embedding model and dimension
        self.embedding_model = embedding_model
        if dimension is None and embedding_model is not None:
            self.dimension = embedding_model.get_sentence_embedding_dimension()
        else:
            self.dimension = dimension

        # 3. Handle the index name
        if index_name == "random":
            while True:
                self.index_name = generate_random_collection_name()
                if not self.client.indices.exists(index=self.index_name):
                    break
            logger.info(f"Chonkie will create a new index in Elasticsearch: {self.index_name}")
        else:
            self.index_name = index_name

        # 4. Create the index with the correct vector mapping if it doesn't exist
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "properties": {
                    "embedding": {"type": "dense_vector", "dims": self.dimension},
                    "text": {"type": "text"},
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "token_count": {"type": "integer"},
                }
            }
            self.client.indices.create(index=self.index_name, mappings=mapping)
            logger.info(f"Index '{self.index_name}' created with vector mapping.")

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importlib.util.find_spec("elasticsearch") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Elasticsearch, bulk
            from elasticsearch import Elasticsearch
            from elasticsearch.helpers import bulk
        else:
            raise ImportError(
                "Elasticsearch is not installed. "
                + "Please install it with `pip install chonkie[elastic]`."
            )

    def __repr__(self) -> str:
        """Return the string representation of the ElasticConnection."""
        return f"ElasticConnection(index_name={self.index_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query using KNN search.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The number of top similar chunks to retrieve.

        Returns:
            A list of dictionaries, each containing a similar chunk, its metadata, and similarity score.

        """
        if query is not None:
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            embedding = self.embedding_model.encode(query).tolist()
        if embedding is None:
            raise ValueError("Either query (with embedding_model) or embedding must be provided")

        knn_query = {
            "field": "embedding",
            "query_vector": embedding,
            "k": limit,
            "num_candidates": 100,  # A standard parameter for approximate nearest neighbor search
        }

        results = self.client.search(index=self.index_name, knn=knn_query, size=limit)

        # Format the results to match the unified output of other handshakes
        matches = []
        for hit in results["hits"]["hits"]:
            source = hit["_source"]
            matches.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": source.get("text"),
                "start_index": source.get("start_index"),
                "end_index": source.get("end_index"),
                "token_count": source.get("token_count"),
            })
        return matches