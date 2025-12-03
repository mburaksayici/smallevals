"""Pinecone Connection to export Chonkie's Chunks into a Pinecone index."""

import importlib.util as importutil
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import logging
from uuid import NAMESPACE_OID, uuid5

from .base import BaseVDBConnection
from .utils import generate_random_collection_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pinecone
    from sentence_transformers import SentenceTransformer


class PineconeConnection(BaseVDBConnection):
    """Pinecone Connection to export Chonkie's Chunks into a Pinecone index.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
        api_key: str: The Pinecone API key.
        index_name: Union[str, Literal["random"]]: The name of the index to use.
        spec: Optional[pinecone.ServerlessSpec]: The pinecone ServerlessSpec to use for the index. If not provided, will use the default spec.
        embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
        embed: Optional[Dict[str, str]]: The Pinecone integrated embedding model to use. If not provided, will use `embedding_model` to create a new index.
        **kwargs: Additional keyword arguments to pass to the Pinecone client.

    """

    def __init__(
        self,
        client: Optional["pinecone.Pinecone"] = None,
        api_key: Optional[str] = None,
        index_name: Union[str, Literal["random"]] = "random",
        spec: Optional["pinecone.ServerlessSpec"] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        dimension: Optional[int] = None,
        embed: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Pinecone handshake.

        Args:
            client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
            api_key: The Pinecone API key.
            index_name: The name of the index to use, or "random" for auto-generated name.
            spec: Optional[pinecone.ServerlessSpec]: The spec to use for the index. If not provided, will use the default spec.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, can be used for query encoding.
            dimension: Optional[int]: The dimension of the vectors. Required if embed is not provided and embedding_model is not provided.
            embed: The Pinecone integrated embedding model to use.
            **kwargs: Additional keyword arguments to pass to the Pinecone client.

        """
        super().__init__()
        self._import_dependencies()

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.getenv("PINECONE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Pinecone API key is not set. Please provide it as an argument or set the PINECONE_API_KEY environment variable."
                )
            self.client = pinecone.Pinecone(api_key=api_key, source_tag="chonkie")

        self.embed: Optional[Dict[str, str]] = embed
        self.embedding_model = embedding_model
        if embed is not None:
            self.dimension = None
            self.metric = "cosine"
        else:
            if dimension is not None:
                self.dimension = dimension
            elif embedding_model is not None:
                self.dimension = embedding_model.get_sentence_embedding_dimension()
            else:
                raise ValueError("Either embed, embedding_model, or dimension must be provided")
            self.metric = "cosine"

        if index_name == "random":
            while True:
                self.index_name = generate_random_collection_name()
                if not self.client.has_index(self.index_name):
                    break
            logger.info(f"Chonkie created a new index in Pinecone: {self.index_name}")
        else:
            self.index_name = index_name

        # set default value for specs field if not present
        self.spec = spec or pinecone.ServerlessSpec(cloud="aws", region="us-east-1")  # type: ignore

        # Create the index if it doesn't exist
        if not self.client.has_index(self.index_name):
            if self.embed is not None:
                self.client.create_index(  # type: ignore[call-arg]
                    name=self.index_name, spec=self.spec, embed=self.embed, **kwargs
                )
            else:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.spec,
                    **kwargs,
                )
        self.index = self.client.Index(self.index_name)

    def _is_available(self) -> bool:
        return importutil.find_spec("pinecone") is not None

    def _import_dependencies(self) -> None:
        if self._is_available():
            global pinecone
            import pinecone
        else:
            raise ImportError(
                "Pinecone is not installed. Please install it with `pip install chonkie[pinecone]`."
            )


    def __repr__(self) -> str:
        """Return a string representation of the PineconeConnection instance.

        Returns:
            str: A string representation containing the index name.

        """
        return f"PineconeConnection(index_name={self.index_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the Pinecone index for similar chunks.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the matching chunks and their metadata.

        """
        logger.debug(f"Searching Pinecone index: {self.index_name} with limit={top_k}")
        if self.embed is not None:
            # Use Pinecone's integrated embedding model
            results = self.index.query(query=query, top_k=top_k, include_metadata=True)
        elif query is None and embedding is None:
            raise ValueError(
                "Query string or embedding must be provided when using a custom embedding model."
            )
        elif query is not None:
            # warning if both query and embedding are provided, query is used
            if embedding is not None:
                logger.warning("Both query and embedding provided. Using query.")
            # Use custom embedding model to embed the query
            if self.embedding_model is None:
                raise ValueError("embedding_model must be provided to encode query strings")
            embedding = self.embedding_model.encode(query).tolist()  # type: ignore
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)

        matches = []
        for match in results.get("matches", []):
            matches.append({
                "id": match.get("id"),
                "score": match.get("score"),
                **match.get("metadata", {}),
            })
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
